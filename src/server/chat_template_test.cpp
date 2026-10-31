/// \file chat_template_test.cpp
/// \brief The chat template's Jinja dialect: the tests a shipped template uses.
///
/// A chat template is data that arrives with the model, so the engine has to
/// support the dialect real templates are written in. Qwen3.x uses
/// `enable_thinking is undefined`, and minja's `is` operator implemented
/// `defined` but not `undefined`, so EVERY request to such a model was refused
/// with a well-formed 400 ("chat template rejected the request: Unknown type for
/// 'is' operator: undefined"). The server was up and the message was valid JSON,
/// which is why an end-to-end smoke test would not have found it: the model
/// could not answer anything at all.
///
/// These tests render through minja::chat_template, the same path the server
/// uses, so they fail for the reason the server failed rather than for a
/// re-implementation's reason.
///
/// No device, no weights, no network.
///
///   ctest --test-dir build -R OPEN-CHAT-TEMPLATE
#include <cstdio>
#include <string>

#include "minja/chat-template.hpp"

static int failures = 0;
static int checks = 0;

static void ok(bool cond, const std::string& what) {
    ++checks;
    if (cond) {
        std::printf("ok    %s\n", what.c_str());
    } else {
        ++failures;
        std::printf("FAIL  %s\n", what.c_str());
    }
}

static std::string render(const std::string& tmpl, const std::string& user,
                          bool with_thinking_flag, const char* null_name = nullptr) {
    minja::chat_template t(tmpl, "", "");
    minja::chat_template_inputs in;
    in.messages = nlohmann::ordered_json::array(
        {nlohmann::ordered_json{{"role", "user"}, {"content", user}}});
    if (with_thinking_flag) in.extra_context["enable_thinking"] = true;
    if (null_name) in.extra_context[null_name] = nullptr;
    minja::chat_template_options opts;
    opts.apply_polyfills = false;
    return t.apply(in, opts);
}

// ---------------------------------------------------------------------------
// The Qwen3.x idiom: "is undefined or is true". The flag is absent in the
// ordinary chat request, which is the case that used to throw.
// ---------------------------------------------------------------------------
static void test_undefined_with_flag_absent() {
    const std::string tmpl =
        "{%- if enable_thinking is undefined or enable_thinking is true %}"
        "THINK{{ messages[0].content }}"
        "{%- else %}PLAIN{{ messages[0].content }}{%- endif %}";
    try {
        const std::string out = render(tmpl, "hi", false);
        ok(out == "THINKhi", "`is undefined` renders with the flag absent (got \"" + out + "\")");
    } catch (const std::exception& e) {
        ok(false, std::string("`is undefined` renders with the flag absent: threw ") + e.what());
    }
}

static void test_undefined_with_flag_present() {
    const std::string tmpl =
        "{%- if enable_thinking is undefined or enable_thinking is true %}"
        "THINK{{ messages[0].content }}"
        "{%- else %}PLAIN{{ messages[0].content }}{%- endif %}";
    try {
        const std::string out = render(tmpl, "hi", true);
        ok(out == "THINKhi", "`is undefined or is true` renders with the flag true");
    } catch (const std::exception& e) {
        ok(false, std::string("`is undefined or is true`: threw ") + e.what());
    }
}

static void test_undefined_negation() {
    const std::string tmpl = "{%- if y is not undefined %}HAS{%- else %}NONE{%- endif %}";
    try {
        ok(render(tmpl, "hi", false) == "NONE", "`is not undefined` is true for an absent name");
    } catch (const std::exception& e) {
        ok(false, std::string("`is not undefined`: threw ") + e.what());
    }
}

// ---------------------------------------------------------------------------
// `defined` must keep meaning what it meant. minja reads a name that is present
// but null as defined, and `undefined` is NOT its exact mirror in that case --
// pinning it keeps the new test from silently changing the old one.
// ---------------------------------------------------------------------------
static void test_defined_still_works() {
    // minja reads a present-but-null name as NOT defined (`!is_null()`), which is
    // a simplification of Jinja's "the name exists" and NOT the same question
    // `undefined` answers. Widening `defined` would change every template that
    // already relies on it, so this test pins the existing meaning rather than
    // the one Jinja documents.
    const std::string tmpl = "{%- if x is defined %}D{%- else %}U{%- endif %}";
    try {
        ok(render(tmpl, "hi", false, "x") == "U",
           "`is defined` still reads a present-but-null name as not defined (minja's meaning)");
    } catch (const std::exception& e) {
        ok(false, std::string("`is defined`: threw ") + e.what());
    }
    try {
        ok(render("{%- if nope is defined %}D{%- else %}U{%- endif %}", "hi", false) == "U",
           "`is defined` is false for an absent name");
    } catch (const std::exception& e) {
        ok(false, std::string("`is defined` (absent): threw ") + e.what());
    }
}

// ---------------------------------------------------------------------------
// The real shipped template, rendered for an ordinary request. This is the test
// that would have caught the outage.
// ---------------------------------------------------------------------------
static void test_shipped_template(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        ok(false, "shipped chat template opens: " + path);
        return;
    }
    std::string tmpl;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) tmpl.append(buf, n);
    std::fclose(f);
    try {
        const std::string out = render(tmpl, "hello", false);
        ok(!out.empty(), "the shipped template renders an ordinary request");
        ok(out.find("hello") != std::string::npos, "the rendered prompt carries the user text");
    } catch (const std::exception& e) {
        ok(false, std::string("the shipped template renders an ordinary request: threw ") +
                      e.what());
    }
}

int main(int argc, char** argv) {
    test_undefined_with_flag_absent();
    test_undefined_with_flag_present();
    test_undefined_negation();
    test_defined_still_works();
    if (argc > 1) test_shipped_template(argv[1]);

    std::printf("\n%s (%d checks, %d failures)\n", failures ? "FAILED" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
