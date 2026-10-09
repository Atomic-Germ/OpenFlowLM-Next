// Traces: SERVER-IMAGES-EDITS (canonical spec: specs/server-api/spec.md)
//
// multipart_test: parse_multipart on hand-built bodies. No device, no network.
#include <cstdio>
#include <string>
#include <vector>

#include "multipart.hpp"

static int failures = 0;
static int checks = 0;

static void ok(bool cond, const std::string& what) {
    ++checks;
    if (!cond) ++failures;
    std::printf("%s  %s\n", cond ? "ok  " : "FAIL", what.c_str());
}

static void eq(const std::string& got, const std::string& want, const std::string& what) {
    ok(got == want, what + (got == want ? "" : "  (got \"" + got + "\", want \"" + want + "\")"));
}

static std::multimap<std::string, MultipartPart> parse(const std::string& boundary_param,
                                                       const std::string& boundary,
                                                       const std::vector<std::string>& parts) {
    http::request<http::string_body> req;
    req.set(http::field::content_type, "multipart/form-data; boundary=" + boundary_param);
    std::string body;
    for (const auto& p : parts) body += "--" + boundary + "\r\n" + p + "\r\n";
    req.body() = body + "--" + boundary + "--\r\n";
    return parse_multipart(req);
}

static std::string names(const std::multimap<std::string, MultipartPart>& m) {
    std::string s;
    for (const auto& [k, _] : m) s += (s.empty() ? "" : ",") + k;
    return s;
}

int main() {
    const std::string B = "XyZ";
    auto one = [&](const std::string& headers) {
        return parse(B, B, {headers + "\r\n\r\nPNGDATA"});
    };

    {
        auto m = parse(B, B, {"Content-Disposition: form-data; name=\"prompt\"\r\n\r\na fox",
                              "Content-Disposition: form-data; name=\"image\"; filename=\"fox.png\"\r\n"
                              "Content-Type: image/png\r\n\r\nPNGDATA"});
        eq(names(m), "image,prompt", "two parts, by name");
        eq(m.find("prompt")->second.content, "a fox", "a field's content");
        eq(m.find("image")->second.filename, "fox.png", "a file's filename");
        eq(m.find("image")->second.content_type, "image/png", "a file's content type");
        eq(m.find("image")->second.content, "PNGDATA", "a file's bytes");
    }
    eq(names(parse("\"" + B + "\"", B, {"Content-Disposition: form-data; name=\"n\"\r\n\r\n2"})), "n",
       "a quoted boundary");
    eq(names(one("CONTENT-DISPOSITION: form-data; NAME=\"image\"")), "image",
       "header and parameter names in any case");
    eq(names(parse(B, B, {"Content-Disposition: form-data; name=\"image[]\"\r\n\r\na",
                          "Content-Disposition: form-data; name=\"image[]\"\r\n\r\nb"})),
       "image[],image[]", "a repeated part is kept twice");

    eq(one("Content-Disposition: form-data; filename=\"a.png\"; name=\"image\"").begin()->first, "image",
       "filename= before name= does not answer for name");
    eq(names(one("Content-Disposition: form-data; filename=\"a.png\"\r\nContent-Type: image/png; name=\"mask\"")),
       "", "name= on another header does not name a part that has none");
    eq(names(one("Content-Type: image/png; name=\"mask\"\r\nContent-Disposition: form-data; name=\"image\"")),
       "image", "...nor rename one that has one");
    eq(names(one("X-Note: content-disposition: form-data; name=\"mask\"\r\n"
                 "Content-Disposition: form-data; name=\"image\"")),
       "image", "a header whose value mentions content-disposition is not it");
    eq(one("X-Content-Type: text/plain\r\nContent-Disposition: form-data; name=\"image\"")
           .begin()->second.content_type,
       "", "x-content-type is not content-type");
    {
        auto m = one("Content-Disposition: form-data; filename=\"x; name=mask\"; name=\"image\"");
        eq(names(m), "image", "name= inside a quoted filename is not a parameter");
        eq(m.begin()->second.filename, "x; name=mask", "...and the filename keeps it");
    }
    {
        auto m = one("Content-Disposition: form-data; filename=\"a\\\"; name=\\\"mask\"; name=\"image\"");
        eq(names(m), "image", "an escaped quote does not end a quoted filename");
        eq(m.begin()->second.filename, "a\"; name=\"mask", "...and is unescaped");
    }
    eq(one("Content-Disposition: form-data; name=\"image\"; filename=\"C:\\dir\\fox.png\"").begin()->second.filename,
       "C:\\dir\\fox.png", "a Windows path's backslashes are kept");
    eq(names(one("Content-Disposition: form-data; name=image")), "image", "an unquoted name");

    std::printf("\n%s (%d checks, %d failures)\n", failures ? "FAILED" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
