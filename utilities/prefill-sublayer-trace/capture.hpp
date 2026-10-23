#pragma once
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
namespace open_qwen36::sublayer_trace {
inline std::string prefix;
inline size_t start=0, real_rows=0;
inline int target(int layer) { return layer==63?7:layer==60?38:layer==35?80:-1; }
inline bool selected(int l,size_t p) { return target(l)>=0 && p==static_cast<size_t>(target(l)); }
inline void save(const std::string& name,int l,size_t p,const void* data,size_t bytes) {
 const auto path=prefix+"_L"+std::to_string(l)+"_P"+std::to_string(p)+"_"+name+".bin";
 if(prefix.empty() || !bytes || !data || std::filesystem::exists(path)) throw std::runtime_error("invalid or existing sublayer capture: "+path);
 std::ofstream f(path,std::ios::binary);f.write(static_cast<const char*>(data),bytes);f.close();
 if(!f)throw std::runtime_error("sublayer write failed: "+path);
}
inline void row(const std::string& name,int l,size_t pos,size_t rows,size_t width,const float* x) {
 int p=target(l);if(p<0 || static_cast<size_t>(p)<pos || static_cast<size_t>(p)-pos>=rows)return;
 if(!width || !x)throw std::runtime_error("invalid sublayer row");
 const float* r=x+(static_cast<size_t>(p)-pos)*width;
 for(size_t i=0;i<width;++i)if(!std::isfinite(r[i]))throw std::runtime_error("non-finite sublayer capture");
 save(name,l,p,r,width*sizeof(float));
}
inline void stage(const std::string& name,int l,size_t width,const float* x){row(name,l,start,real_rows,width,x);}
}
