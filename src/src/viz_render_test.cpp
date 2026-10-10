// Traces: VIZ-PAGE-FILL (canonical spec: specs/viz/spec.md)
#include <cstdio>
#include <string>

#include "viz_render.hpp"

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

static std::string between(const std::string& s, const std::string& open, const std::string& close) {
    size_t a = s.find(open);
    if (a == std::string::npos) return "<missing>";
    a += open.size();
    return s.substr(a, s.find(close, a) - a);
}

int main() {
    const std::string tpl =
        "<style>/*VIZ_FONTS*/</style><script id=d>/*VIZ_DATA*/</script>"
        "<script id=e>/*VIZ_EXPLAIN*/</script><script id=m>/*VIZ_MODEL*/</script>";
    const std::string data = R"({"note":"</script><script>alert(1)</script> /*VIZ_EXPLAIN*/"})";
    const std::string html = viz_command::render(tpl, "@font-face{}", data, "## core:x\nTitle: </b>", R"({"tag":"a:b"})");

    check(html.find("</script><script>alert") == std::string::npos, "a payload cannot close its script tag");
    check(between(html, "<script id=d>", "</script>") == R"({"note":"<\/script><script>alert(1)<\/script> /*VIZ_EXPLAIN*/"})",
          "the data keeps its own slot-looking text, escaped");
    check(between(html, "<script id=e>", "</script>") == "## core:x\nTitle: <\\/b>", "the explainers fill their own slot");
    check(between(html, "<script id=m>", "</script>") == R"({"tag":"a:b"})", "the model entry fills its slot");
    check(between(html, "<style>", "</style>") == "@font-face{}", "the fonts are inlined");

    bool refused = false;
    try {
        viz_command::render("<p>/*VIZ_FONTS*/</p>", "", "", "", "");
    } catch (const std::exception&) {
        refused = true;
    }
    check(refused, "a template missing a slot is refused");

    if (!failures) std::printf("viz_render_test: all passed\n");
    return failures ? 1 : 0;
}
