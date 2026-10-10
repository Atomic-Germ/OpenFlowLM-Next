# cmake -DOUT=<cpp> -DPAGE=<viz.html> -DFONTS=<fonts.css> -DEXPLAIN_DIR=<dir> -P embed.cmake
function(viz_hex var)
  set(hex "")
  foreach(f IN LISTS ARGN)
    file(READ "${f}" h HEX)
    string(APPEND hex "${h}0a0a")
  endforeach()
  string(LENGTH "${hex}" n)
  math(EXPR n "${n} / 2")
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex "${hex}")
  string(REGEX REPLACE "((0x..,)(0x..,)(0x..,)(0x..,)(0x..,)(0x..,)(0x..,)(0x..,)(0x..,)(0x..,)(0x..,)(0x..,)(0x..,)(0x..,)(0x..,)(0x..,))" "\\1\n" hex "${hex}")
  set(${var} "${hex}" PARENT_SCOPE)
  set(${var}_n ${n} PARENT_SCOPE)
endfunction()

file(GLOB md "${EXPLAIN_DIR}/*.md")
list(SORT md)
viz_hex(page "${PAGE}")
viz_hex(fonts "${FONTS}")
viz_hex(explain ${md})
file(WRITE "${OUT}.tmp" "// generated from src/viz by embed.cmake\n#include <cstddef>\nnamespace viz_assets {\n"
  "extern const unsigned char page[] = {\n${page}};\nextern const size_t page_size = ${page_n};\n"
  "extern const unsigned char fonts[] = {\n${fonts}};\nextern const size_t fonts_size = ${fonts_n};\n"
  "extern const unsigned char explain[] = {\n${explain}0};\nextern const size_t explain_size = ${explain_n};\n}\n")
file(COPY_FILE "${OUT}.tmp" "${OUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUT}.tmp")
