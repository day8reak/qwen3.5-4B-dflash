// Temporary single-GDR ACL runner. No Python ACL dependency or profiling API.
#include <acl/acl.h>
#include "qwen35_dflash/sha256.hpp"
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
#ifdef GDR_DEBUG_FAKE_ACL
constexpr const char* kBackend = "host fake ACL TEST ONLY";
constexpr const char* kFallback = "true";
#else
constexpr const char* kBackend = "AscendCL C++";
constexpr const char* kFallback = "false";
#endif
void Require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
void Check(aclError code, const char* call) {
  if (code != ACL_SUCCESS) throw std::runtime_error(std::string(call) + " failed: " + std::to_string(code));
}
#define ACL(call) Check((call), #call)
std::string Quote(const std::string& s) {
  std::ostringstream out; out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  out << '"'; return out.str();
}
std::string Address(void* p) { std::ostringstream s; s << p; return s.str(); }
struct Spec {
  std::string name, file;
  int dtype = -1;
  std::size_t bytes = 0, valid_bytes = 0;
  std::vector<std::int64_t> dims;
};
struct Plan {
  std::string model, output, profile, metrics;
  int device = -1, warmup = 0, repetitions = 0;
  std::vector<Spec> inputs, outputs;
  explicit Plan(const char* path) {
    std::ifstream in(path);
    std::string version; in >> version;
    Require(version == "GDR_DEBUG_V1", "invalid plan version");
    in >> std::quoted(model) >> std::quoted(output) >> device >> warmup >> repetitions
       >> std::quoted(profile) >> std::quoted(metrics);
    Require(device >= 0 && warmup >= 0 && warmup <= 10000 && repetitions > 0 && repetitions <= 10000,
            "invalid invocation counts or device");
    Require(metrics == "PipeUtilization" || metrics == "Memory" || metrics == "MemoryUB", "invalid metrics");
    std::size_t n = 0; in >> n; Require(n == 7, "expected seven GDR inputs");
    const std::vector<std::string> names{"query","key","value","g","beta","initial_state","effective_length"};
    const std::vector<int> types{1,1,1,0,1,0,6};
    const std::vector<std::vector<std::int64_t>> dims{
        {1,16,32,128},{1,16,32,128},{1,16,32,128},{1,16,32},{1,16,32},{1,32,128,128},{1}};
    const std::vector<std::size_t> bytes{131072,131072,131072,2048,1024,2097152,2};
    for (std::size_t i=0; i<n; ++i) {
      Spec s; std::size_t rank = 0;
      in >> std::quoted(s.name) >> s.dtype >> s.bytes >> rank;
      Require(rank == dims[i].size(), "unexpected input rank");
      s.dims.resize(rank); for (auto& d : s.dims) in >> d;
      in >> std::quoted(s.file);
      Require(s.name == names[i] && s.dtype == types[i] && s.bytes == bytes[i] && s.dims == dims[i],
              "invalid GDR input contract: " + s.name);
      Require(fs::file_size(s.file) == s.bytes, "wrong input file bytes: " + s.name);
      inputs.push_back(s);
    }
    std::int16_t length = 0;
    std::ifstream lenfile(inputs.back().file, std::ios::binary);
    lenfile.read(reinterpret_cast<char*>(&length), sizeof(length));
    Require(bool(lenfile) && length >= 1 && length <= 16, "effective_length must be INT16[1] in 1..16");
    in >> n; Require(n == 1 || n == 2, "expected one or two outputs");
    for (std::size_t i=0; i<n; ++i) {
      Spec s; in >> std::quoted(s.name) >> s.dtype >> s.bytes >> s.valid_bytes;
      const bool core = s.name == "core_attn";
      Require(core || s.name == "last_recurrent_state", "unexpected output name");
      Require(s.dtype == (core ? 1 : 0) && s.bytes == (core ? 131072u : 2097152u) &&
              s.valid_bytes == (core ? std::size_t(length)*32*128*2 : 2097152u), "wrong output contract");
      if (n == 2) Require(core == (i == 0), "wrong output ordering");
      outputs.push_back(s);
    }
    Require(bool(in), "truncated plan");
    std::string extra; Require(!(in >> extra), "trailing plan fields");
    const auto run = std::getenv("AI_RUN_DIR");
    Require(run != nullptr, "AI_RUN_DIR is required");
    const auto relative = fs::weakly_canonical(output).lexically_relative(fs::weakly_canonical(run));
    Require(!relative.empty() && relative != "." && *relative.begin() != "..", "output escapes AI_RUN_DIR");
    Require(!fs::exists(output) && !fs::is_symlink(output), "output directory already exists");
    fs::create_directories(output);
  }
};

// Matches the existing models.dflash_v1.msprof_cli acknowledged socket protocol.
class Channel {
  int fd_ = -1;
 public:
  explicit Channel(bool enabled) {
    if (!enabled) return;
    const char* fd = std::getenv("DFLASH_MSPROF_CONTROL_FD");
    const char* mode = std::getenv("PROFILING_MODE");
    Require(fd && mode && std::string(mode) == "dynamic", "use the debug profile subcommand");
    fd_ = std::stoi(fd);
    const char* value = std::getenv("DFLASH_MSPROF_CONTROL_TIMEOUT");
    double seconds = value ? std::stod(value) : 2400;
    Require(std::isfinite(seconds) && seconds > 0 && seconds < 1e8, "invalid controller timeout");
    timeval tv{}; tv.tv_sec = static_cast<long>(seconds); tv.tv_usec = static_cast<long>((seconds-tv.tv_sec)*1e6);
    if (!tv.tv_sec && !tv.tv_usec) tv.tv_usec = 1;
    Require(setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0 &&
            setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0 &&
            fcntl(fd_, F_SETFD, FD_CLOEXEC) == 0, "cannot configure control socket");
    unsetenv("DFLASH_MSPROF_CONTROL_FD");
  }
  ~Channel() { if (fd_ >= 0) close(fd_); }
  void Exchange(const std::string& message, const char* expected) {
    std::string line = message + '\n'; std::size_t offset = 0;
    while (offset < line.size()) {
      auto n = send(fd_, line.data()+offset, line.size()-offset, MSG_NOSIGNAL);
      if (n < 0 && errno == EINTR) continue;
      Require(n > 0, "profile send failed or timed out"); offset += static_cast<std::size_t>(n);
    }
    std::string response;
    for (;;) {
      char c=0; auto n=recv(fd_, &c, 1, 0);
      if (n < 0 && errno == EINTR) continue;
      Require(n == 1, "profile controller disconnected or timed out");
      if (c == '\n') break;
      Require(response.size() < 4096, "oversized controller reply");
      if (c != ' ' && c != '\t' && c != '\r') response += c;
    }
    Require(response == std::string("{\"event\":\"")+expected+"\"}", "unexpected controller reply");
  }
};
struct Buffer {
  Spec spec;
  std::size_t allocated = 0;
  void *host = nullptr, *device = nullptr;
  aclDataBuffer* descriptor = nullptr;
  std::vector<char> original, reference;
  std::string actual_name;
  std::vector<std::int64_t> actual_dims;
  ~Buffer() {
    if (descriptor) aclDestroyDataBuffer(descriptor);
    if (device) aclrtFree(device);
    if (host) aclrtFreeHost(host);
  }
};
class Runtime {
  bool initialized_ = false, device_set_ = false, loaded_ = false;
  int device_id_;
 public:
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  std::uint32_t model = 0;
  aclmdlDesc* desc = nullptr;
  aclmdlDataset *input_set = nullptr, *output_set = nullptr;
  std::vector<std::unique_ptr<Buffer>> inputs, outputs;
  explicit Runtime(int device) : device_id_(device) {}
  ~Runtime() {
    if (stream) aclrtSynchronizeStream(stream);
    if (input_set) aclmdlDestroyDataset(input_set);
    if (output_set) aclmdlDestroyDataset(output_set);
    inputs.clear(); outputs.clear();
    if (desc) aclmdlDestroyDesc(desc);
    if (loaded_) aclmdlUnload(model);
    if (stream) aclrtDestroyStream(stream);
    if (context) aclrtDestroyContext(context);
    if (device_set_) aclrtResetDevice(device_id_);
    if (initialized_) aclFinalize();
  }
  void Init(const Plan& p) {
    ACL(aclInit(nullptr)); initialized_ = true;
    ACL(aclrtSetDevice(device_id_)); device_set_ = true;
    ACL(aclrtCreateContext(&context, device_id_)); ACL(aclrtSetCurrentContext(context));
    ACL(aclrtCreateStream(&stream)); ACL(aclmdlLoadFromFile(p.model.c_str(), &model)); loaded_ = true;
    desc = aclmdlCreateDesc(); Require(desc, "aclmdlCreateDesc failed"); ACL(aclmdlGetDesc(desc, model));
    Require(aclmdlGetNumInputs(desc) == p.inputs.size() && aclmdlGetNumOutputs(desc) == p.outputs.size(),
            "OM input/output count differs from the audited graph");
    input_set = aclmdlCreateDataset(); output_set = aclmdlCreateDataset();
    Require(input_set && output_set, "aclmdlCreateDataset failed");
    Allocate(p.inputs, true); Allocate(p.outputs, false);
  }
  void Allocate(const std::vector<Spec>& specs, bool input) {
    auto& buffers = input ? inputs : outputs;
    for (std::size_t i=0; i<specs.size(); ++i) {
      auto b = std::make_unique<Buffer>(); b->spec = specs[i];
      auto type = input ? aclmdlGetInputDataType(desc,i) : aclmdlGetOutputDataType(desc,i);
      b->allocated = input ? aclmdlGetInputSizeByIndex(desc,i) : aclmdlGetOutputSizeByIndex(desc,i);
      aclmdlIODims dims{};
      if (input) ACL(aclmdlGetInputDims(desc,i,&dims)); else ACL(aclmdlGetOutputDims(desc,i,&dims));
      Require(dims.dimCount > 0 && dims.dimCount <= 8, "invalid OM rank");
      b->actual_dims.assign(dims.dims,dims.dims+dims.dimCount); b->actual_name = dims.name;
      std::size_t elements=1;
      for (auto d : b->actual_dims) {
        Require(d > 0 && d <= 2097152 && elements <= 2097152/std::size_t(d), "invalid OM dimensions");
        elements *= std::size_t(d);
      }
      auto item_size = b->spec.dtype == ACL_FLOAT ? 4u : 2u;
      Require(type == b->spec.dtype && elements*item_size == b->spec.bytes &&
              b->allocated >= b->spec.bytes && b->allocated <= b->spec.bytes+4096,
              "OM dtype/size mismatch at " + b->spec.name + ": dtype=" + std::to_string(type) +
              " allocated=" + std::to_string(b->allocated) + " expected=" + std::to_string(b->spec.bytes));
      Require(!input || b->actual_dims == b->spec.dims, "OM input shape mismatch at " + b->spec.name);
      ACL(aclrtMallocHost(&b->host,b->allocated));
      ACL(aclrtMalloc(&b->device,b->allocated,ACL_MEM_MALLOC_NORMAL_ONLY));
      b->descriptor = aclCreateDataBuffer(b->device,b->allocated);
      Require(b->descriptor, "aclCreateDataBuffer failed");
      ACL(aclmdlAddDatasetBuffer(input ? input_set : output_set,b->descriptor));
      if (input) {
        b->original.resize(b->spec.bytes);
        std::ifstream file(b->spec.file,std::ios::binary);
        file.read(b->original.data(),static_cast<std::streamsize>(b->original.size()));
        Require(bool(file), "cannot read input: " + b->spec.file);
      }
      buffers.push_back(std::move(b));
    }
  }
  void Reset() {
    for (auto& b : inputs) {
      std::memset(b->host,0,b->allocated);
      std::memcpy(b->host,b->original.data(),b->original.size());
      ACL(aclrtMemcpyAsync(b->device,b->allocated,b->host,b->allocated,ACL_MEMCPY_HOST_TO_DEVICE,stream));
    }
    ACL(aclrtSynchronizeStream(stream));
  }
  void Execute() {
    ACL(aclmdlExecuteAsync(model,input_set,output_set,stream));
    ACL(aclrtSynchronizeStream(stream));
  }
  bool Read(bool measured, const Plan& p, std::string& hashes) {
    for (auto* group : {&inputs,&outputs}) for (auto& b : *group)
      ACL(aclrtMemcpyAsync(b->host,b->allocated,b->device,b->allocated,ACL_MEMCPY_DEVICE_TO_HOST,stream));
    ACL(aclrtSynchronizeStream(stream));
    for (auto& b : inputs)
      Require(std::memcmp(b->host,b->original.data(),b->original.size()) == 0,
              "GDR modified declared read-only input: " + b->spec.name);
    bool stable = true; hashes = "{";
    for (std::size_t i=0; i<outputs.size(); ++i) {
      auto& b = outputs[i];
      const auto* bytes = static_cast<char*>(b->host);
      if (measured) {
        if (b->reference.empty()) b->reference.assign(bytes,bytes+b->spec.valid_bytes);
        stable = stable && std::memcmp(bytes,b->reference.data(),b->spec.valid_bytes) == 0;
        if (i) hashes += ",";
        hashes += Quote(b->spec.name)+":"+Quote(qwen35::dflash::Sha256(std::string_view(bytes,b->spec.valid_bytes)));
        std::ofstream out(fs::path(p.output)/(b->spec.name+".bin"),std::ios::binary);
        out.write(bytes,static_cast<std::streamsize>(b->spec.bytes)); Require(bool(out), "output write failed");
      }
    }
    hashes += "}"; return stable;
  }
  std::string Metadata() const {
    std::ostringstream s; s << '['; bool first=true;
    for (auto* group : {&inputs,&outputs}) for (const auto& b : *group) {
      if (!first) s << ',';
      first = false;
      s << "{\"name\":" << Quote(b->spec.name) << ",\"direction\":" << Quote(group == &inputs ? "input" : "output")
        << ",\"om_name\":" << Quote(b->actual_name) << ",\"dtype\":" << b->spec.dtype
        << ",\"logical_bytes\":" << b->spec.bytes << ",\"allocated_bytes\":" << b->allocated
        << ",\"device_address\":" << Quote(Address(b->device)) << ",\"shape\":[";
      for (std::size_t i=0;i<b->actual_dims.size();++i) { if(i) s << ','; s << b->actual_dims[i]; }
      s << "]}";
    }
    s << ']'; return s.str();
  }
};

int main(int argc,char** argv) {
  try {
    Require(argc == 2, "usage: gdr_debug_runner PLAN.txt");
    const auto simulation=std::getenv("ASCEND310P_SIMULATION_ONLY");
    Require(!simulation || std::string(simulation)!="1", "simulation-only profiles cannot measure device performance");
    Plan p(argv[1]); Runtime runtime(p.device); runtime.Init(p);
    const bool profiled = !p.profile.empty(); Channel channel(profiled);
    std::ostringstream samples; samples << std::setprecision(17) << '[';
    bool all_stable=true;
    const int measured_calls = profiled ? 1 : p.repetitions;
    for (int i=0;i<p.warmup+measured_calls;++i) {
      runtime.Reset(); const bool measured=i>=p.warmup;
      if(profiled && measured) channel.Exchange("{\"event\":\"ready\",\"pid\":"+std::to_string(getpid())+
          ",\"stage\":\"verify\",\"output\":"+Quote(p.profile)+",\"device_id\":"+std::to_string(p.device)+
          ",\"metrics\":"+Quote(p.metrics)+"}","started");
      auto start=Clock::now();
      try { runtime.Execute(); }
      catch (...) {
        if(profiled && measured) channel.Exchange("{\"event\":\"done\",\"success\":false}","stopped");
        throw;
      }
      const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
      if(profiled && measured) channel.Exchange("{\"event\":\"done\",\"success\":true}","stopped");
      std::string hashes; bool stable=runtime.Read(measured,p,hashes); all_stable=all_stable&&stable;
      if(measured) {
        if(i>p.warmup) samples << ',';
        samples << "{\"elapsed_ms\":" << ms << ",\"stable\":" << (stable?"true":"false")
                << ",\"valid_output_sha256\":" << hashes << '}';
      }
    }
    samples << ']';
    std::ofstream report(fs::path(p.output)/"report.json");
    report << "{\"backend\":" << Quote(kBackend) << ",\"device_id\":" << p.device << ",\"cpu_fallback\":" << kFallback << ','
           << "\"warmup\":" << p.warmup << ",\"profiled\":" << (profiled?"true":"false")
           << ",\"stable\":" << (all_stable?"true":"false") << ",\"samples\":" << samples.str()
           << ",\"model_sha256\":" << Quote(qwen35::dflash::Sha256File(p.model))
           << ",\"tensors\":" << runtime.Metadata()
           << ",\"timing_scope\":\"aclmdlExecuteAsync plus stream synchronization; persistent buffers; transfers/checks excluded\"}\n";
    Require(bool(report),"report write failed"); return 0;
  } catch (const std::exception& e) {
    std::cerr << "gdr_debug_runner: " << e.what() << '\n'; return 1;
  }
}
