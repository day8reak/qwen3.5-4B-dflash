// Host-only fixture. Copies inputs to outputs; it does NOT implement GDR.
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
struct aclDataBuffer {void* ptr; std::size_t size;};
struct aclmdlDataset {std::vector<aclDataBuffer*> buffers;};
struct aclmdlDesc {};
static std::string variant;
static int calls = 0;
static bool draft_probe = false;
static std::int64_t probe_input_width = 0;
static std::vector<std::int64_t> probe_output_widths;
static const std::array<std::vector<std::int64_t>,7> shapes{{
    {1,16,32,128},{1,16,32,128},{1,16,32,128},{1,16,32},{1,16,32},{1,32,128,128},{1}}};
static const std::array<std::size_t,7> sizes{{131072,131072,131072,2048,1024,2097152,2}};
static const std::array<aclDataType,7> types{{ACL_FLOAT16,ACL_FLOAT16,ACL_FLOAT16,ACL_FLOAT,ACL_FLOAT16,ACL_FLOAT,ACL_INT16}};
static bool flag(const char* name) {return std::getenv(name) != nullptr;}
static bool core(std::size_t i) {return variant != "state" && i == 0;}
aclError aclInit(const char*){calls=0; return 0;}
aclError aclFinalize(){return 0;}
aclError aclrtSetDevice(int){return 0;}
aclError aclrtResetDevice(int){return 0;}
aclError aclrtCreateContext(aclrtContext* x,int){*x=reinterpret_cast<void*>(1);return 0;}
aclError aclrtDestroyContext(aclrtContext){return 0;}
aclError aclrtSetCurrentContext(aclrtContext){return 0;}
aclError aclrtCreateStream(aclrtStream* x){*x=reinterpret_cast<void*>(1);return 0;}
aclError aclrtDestroyStream(aclrtStream){return 0;}
aclError aclrtSynchronizeStream(aclrtStream){return 0;}
aclError aclrtMallocHost(void** p,std::size_t n){*p=std::malloc(n);return *p?0:1;}
aclError aclrtFreeHost(void* p){std::free(p);return 0;}
aclError aclrtMalloc(void** p,std::size_t n,aclrtMemMallocPolicy){*p=std::malloc(n);return *p?0:1;}
aclError aclrtFree(void* p){std::free(p);return 0;}
aclError aclrtMemcpyAsync(void* dst,std::size_t max,const void* src,std::size_t n,aclrtMemcpyKind,aclrtStream){
  if(n>max)return 1;
  std::memcpy(dst,src,n);return 0;
}
aclError aclmdlLoadFromFile(const char* path,std::uint32_t* id){
  std::ifstream file(path); std::string magic; file >> magic;
  draft_probe = magic == "FAKE_DRAFT_PROBE";
  if (draft_probe) {
    std::size_t count = 0; file >> probe_input_width >> count;
    probe_output_widths.resize(count); for(auto& width:probe_output_widths) file >> width;
    if (!file || count < 1 || count > 3) return 1;
    *id=1;return 0;
  }
  std::string s(path);variant=s.find("gdr_state")!=std::string::npos?"state":s.find("gdr_core")!=std::string::npos?"core":"both";
  *id=1;return 0;
}
aclError aclmdlUnload(std::uint32_t){return 0;}
aclmdlDesc* aclmdlCreateDesc(){return new aclmdlDesc;}
aclError aclmdlDestroyDesc(aclmdlDesc* x){delete x;return 0;}
aclError aclmdlGetDesc(aclmdlDesc*,std::uint32_t){return 0;}
std::size_t aclmdlGetNumInputs(const aclmdlDesc*){return draft_probe?1:7;}
std::size_t aclmdlGetNumOutputs(const aclmdlDesc*){return draft_probe?probe_output_widths.size():(variant=="both"?2:1);}
aclError aclmdlGetInputDims(const aclmdlDesc*,std::size_t i,aclmdlIODims* d){
  if(draft_probe){d->dimCount=3;d->dims[0]=1;d->dims[1]=64;d->dims[2]=probe_input_width;return 0;}
  d->dimCount=shapes[i].size();std::copy(shapes[i].begin(),shapes[i].end(),d->dims);return 0;
}
aclError aclmdlGetOutputDims(const aclmdlDesc*,std::size_t i,aclmdlIODims* d){
  if(draft_probe){d->dimCount=3;d->dims[0]=1;d->dims[1]=64;d->dims[2]=probe_output_widths[i];return 0;}
  if(core(i)){d->dimCount=2; d->dims[0]=512;d->dims[1]=128;}
  else{d->dimCount=4;std::copy(shapes[5].begin(),shapes[5].end(),d->dims);}return 0;
}
aclDataType aclmdlGetInputDataType(const aclmdlDesc*,std::size_t i){return draft_probe?ACL_FLOAT16:(flag("GDR_TEST_BAD_DTYPE")&&i==6?ACL_INT32:types[i]);}
aclDataType aclmdlGetOutputDataType(const aclmdlDesc*,std::size_t i){return draft_probe?ACL_FLOAT16:(core(i)?ACL_FLOAT16:ACL_FLOAT);}
std::size_t aclmdlGetInputSizeByIndex(aclmdlDesc*,std::size_t i){return draft_probe?probe_input_width*128:(i==6?32:sizes[i]);}
std::size_t aclmdlGetOutputSizeByIndex(aclmdlDesc*,std::size_t i){return (draft_probe?probe_output_widths[i]*128:(core(i)?131072:2097152))+32;}
aclmdlDataset* aclmdlCreateDataset(){return new aclmdlDataset;}
aclError aclmdlDestroyDataset(aclmdlDataset* x){delete x;return 0;}
aclDataBuffer* aclCreateDataBuffer(void* p,std::size_t n){return new aclDataBuffer{p,n};}
aclError aclDestroyDataBuffer(aclDataBuffer* x){delete x;return 0;}
aclError aclmdlAddDatasetBuffer(aclmdlDataset* d,aclDataBuffer* b){d->buffers.push_back(b);return 0;}
aclError aclmdlExecuteAsync(std::uint32_t,const aclmdlDataset* in,aclmdlDataset* out,aclrtStream){
  ++calls;
  if(draft_probe){
    for(std::size_t i=0;i<out->buffers.size();++i){
      auto* b=out->buffers[i];std::memset(b->ptr,0,b->size);
      std::memcpy(b->ptr,in->buffers[0]->ptr,std::min(in->buffers[0]->size,probe_output_widths[i]*128ul));
      if(flag("PROBE_TEST_UNSTABLE")&&calls==2)static_cast<char*>(b->ptr)[2]^=1;
      if(flag("PROBE_TEST_PADDING_UNSTABLE")&&calls==2)
        static_cast<char*>(b->ptr)[probe_output_widths[i]*128-2]^=1;
    }
    if(flag("GDR_TEST_MUTATE"))static_cast<char*>(in->buffers[0]->ptr)[0]^=1;
    return 0;
  }
  std::int16_t length=0;std::memcpy(&length,in->buffers[6]->ptr,2);
  if(length<1||length>16)return 1;
  for(std::size_t i=0;i<out->buffers.size();++i){
    auto* b=out->buffers[i];auto* src=in->buffers[core(i)?0:5];
    auto bytes=core(i)?131072u:2097152u;std::memcpy(b->ptr,src->ptr,bytes);
    if(core(i)&&length<16)std::memset(static_cast<char*>(b->ptr)+std::size_t(length)*8192,calls,(16-length)*8192);
    if(flag("GDR_TEST_UNSTABLE"))static_cast<char*>(b->ptr)[0]=static_cast<char>(calls);
  }
  if(flag("GDR_TEST_MUTATE"))static_cast<char*>(in->buffers[5]->ptr)[0]^=1;
  return 0;
}
