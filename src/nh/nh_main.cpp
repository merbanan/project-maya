// src/nh/nh_main.cpp - maya-nh: NVIDIA-Nemotron-3.5 (Nemotron-H, EXL3) on one NVIDIA GPU.
//
//   maya-nh --pack <model>/maya-nh --serve [options]     Maya's engine protocol on stdin / stdout
//
// --serve speaks what serve/server.py's StrataEngine expects (maya-mm3's protocol, one conversation): INFO lines, then
// `READY <context> stop`; per request `GEN <max_new> [key=value ...] <id,id,...>` answered by `PP <read> <total> <ms>
// <tok/s> <reused>` as the prompt is read, `T <id>` lines, `STAT ...` every 8 tokens and `DONE <generated> <prompt>
// <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused> <hits> <lookups>`; `STOP` ends a request early, `QUIT` the
// engine.  The Mamba layers' state cannot go back to an earlier position, so a prompt reuses the state only where it
// extends the state's tokens, or from the snapshot taken where the last prompt ended (a chat's next turn re-renders
// the answer after it) - else it is read from the start.
//
//   maya-nh --pack <model>/maya-nh --ids @prompt.txt [--gen N] [--dump-logits out.bin --last N]
//           [--gpu G] [--max-context N] [--chunk N] [--staging N] [--reserve-gb X] [--resident-cap N]
//           [--profile FILE] [--profile-out FILE] [--no-pin-pool] [--split N]
//
// --ids: comma- or whitespace-separated token ids, or @FILE holding them.  --gen: greedy tokens after the prompt
// (their ids printed).  --dump-logits: the logits of the prompt's last --last positions (default 1), as
//   "NHL1" int32 rows, int32 vocab, int32 n_ids, int32 ids[n_ids], float logits[rows][vocab]
// (tools/nh/compare.py reads it beside tools/nh/ref.py's dump).  --split N: the prompt in calls of N tokens (the state
// carried from one to the next - the same logits as one call, within the chunking's rounding).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mm3_os.hpp"
#include "mm3_sample.hpp"
#include "nh_model.hpp"

namespace {

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool read_ids(const std::string& a, std::vector<int>& ids) {
    std::string s = a;
    if (!a.empty() && a[0] == '@') {
        std::ifstream f(a.substr(1));
        if (!f) return false;
        std::stringstream b;
        b << f.rdbuf();
        s = b.str();
    }
    for (char& c : s)
        if (c == ',') c = ' ';
    std::istringstream is(s);
    for (int v; is >> v;) ids.push_back(v);
    return !ids.empty();
}

// stdin's lines on a thread of their own: STOP is seen while a request runs
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::atomic<bool> wrap{false};   // (WRAP: the server is stopping - a request still thinking closes it and answers)
    void start() {
        std::thread([this] {
            std::string buf;
            char chunk[4096];
            for (;;) {
                const long n = mm3::os::read_stdin(chunk, sizeof chunk);
                if (n <= 0) break;
                buf.append(chunk, (size_t) n);
                for (size_t nl; (nl = buf.find('\n')) != std::string::npos;) {
                    std::string l = buf.substr(0, nl);
                    buf.erase(0, nl + 1);
                    if (!l.empty() && l.back() == '\r') l.pop_back();
                    if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                        stop.store(true);
                        continue;
                    }
                    if (l == "WRAP") {
                        wrap.store(true);
                        continue;
                    }
                    std::lock_guard<std::mutex> lk(mu);
                    q.push_back(std::move(l));
                    cv.notify_one();
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        }).detach();
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

using Req = mm3::sample::Req;

bool parse_gen(const std::string& line, Req& r) {
    std::istringstream is(line.substr(4));
    if (!(is >> r.max_new)) return false;
    for (std::string tok; is >> tok;) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            read_ids(tok, r.ids);
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "penalty_last_n") r.pen_n = std::atoi(v.c_str());
        else if (k == "penalty_repeat") r.pen_repeat = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_freq") r.pen_freq = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_present") r.pen_present = std::strtof(v.c_str(), nullptr);
        else if (k == "think_budget") r.think_budget = std::atoll(v.c_str());
        else if (k == "think_end") r.think_end = std::atoi(v.c_str());
        // (keys this engine does not use are taken and ignored)
    }
    return !r.ids.empty();
}

int serve(nh::Model& m, int chunk) {
#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    const nh::Stats& s0 = m.stats();
    for (const std::string& l : m.info_lines()) std::fprintf(stderr, "%s\n", l.c_str());
    std::printf("INFO engine=nh engine_kind=nh model=NVIDIA-Nemotron-3.5 experts_vram=%ld experts=%ld vision=off\n",
                s0.resident, s0.experts);
    std::printf("READY %d stop\n", m.max_ctx());
    std::fflush(stdout);
    std::fprintf(stderr, "maya nh serve: GEN <max_new> [keys] <ids>; STOP, QUIT\n");
    Lines in;
    in.start();
    std::vector<int> state;     // the tokens in the cache, positions 0 ..
    std::vector<int> snap_ids;  // the tokens of the snapshot's state
    std::vector<float> lg((size_t) m.vocab());
    const std::vector<int> eos = m.eos();
    unsigned long long counter = 0;
    std::string line, err;
    const auto refuse = [&](const std::string& msg, size_t n) {
        std::printf("ERR %s\n", msg.c_str());
        std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", n);
        std::fflush(stdout);
    };
    while (in.next(line)) {
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, STOP and QUIT)\n", line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        Req r;
        if (!parse_gen(line, r)) {
            refuse("expected: GEN <max_new> [keys] <id,id,...>", r.ids.size());
            continue;
        }
        if ((long long) r.ids.size() + std::max(0LL, r.max_new) > m.max_ctx()) {
            refuse("the prompt (" + std::to_string(r.ids.size()) + " tokens) and max_new " +
                       std::to_string(r.max_new) + " are more than the context " + std::to_string(m.max_ctx()),
                   r.ids.size());
            continue;
        }
        const double t0 = now() * 1e3;
        // the state's shared start (the last token is always read again: its logits are this request's first)
        size_t common = 0;
        while (common < state.size() && common + 1 < r.ids.size() && state[common] == r.ids[common]) ++common;
        size_t from = 0;
        if (common == state.size() && common > 0) {
            from = common;   // (the request extends the state)
        } else if (m.snapshot_pos() > 0 && (size_t) m.snapshot_pos() <= common &&
                   (size_t) m.snapshot_pos() <= snap_ids.size()) {
            if (!m.restore(err)) {
                refuse(err, r.ids.size());
                state.clear();
                continue;
            }
            from = (size_t) m.snapshot_pos();
        }
        state.resize(from);
        bool ok = true, cancel = false;
        for (size_t a = from; a < r.ids.size();) {
            const int T = (int) std::min<size_t>((size_t) chunk, r.ids.size() - a);
            const bool last = a + (size_t) T == r.ids.size();
            if (!m.forward(r.ids.data() + a, T, (int) a, last ? 1 : 0, last ? lg.data() : nullptr, err)) {
                ok = false;
                break;
            }
            state.insert(state.end(), r.ids.begin() + (long) a, r.ids.begin() + (long) a + T);
            a += (size_t) T;
            const double ms = now() * 1e3 - t0;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", a, r.ids.size(), ms,
                        ms > 0 ? 1000.0 * (double) (a - from) / ms : 0.0, from);
            std::fflush(stdout);
            if (in.stop.load() && !last) {
                cancel = true;
                break;
            }
        }
        const double t1 = now() * 1e3;
        if (!ok) {
            refuse(err, r.ids.size());
            state.clear();
            continue;
        }
        if (cancel) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", r.ids.size(), t1 - t0, from);
            std::fflush(stdout);
            continue;
        }
        // the snapshot where this prompt ends: the next turn's prompt re-renders the answer after it
        if (m.snapshot(err)) snap_ids = state;
        ++counter;
        std::mt19937_64 rng(r.seed != 0 ? r.seed : 0x9E3779B97F4A7C15ull * counter);
        const nh::Stats s_dec = m.stats();
        long long produced = 0;
        const char* finish = "length";
        bool in_think = r.think_budget > 0 && r.think_end >= 0;
        int tok = mm3::sample::draw(mm3::sample::dist(lg.data(), m.vocab(), r, state), rng);
        auto stat_line = [&]() {
            const nh::Stats& st = m.stats();
            const double ms = now() * 1e3 - t1;
            const double per = produced > 0 ? 1.0 / (double) produced : 0.0;
            std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=1.0 cpu=0 vram_used=%ld ram_used=0 vram_gb=%.2f "
                        "ram_gb=%.2f pcie=%.3f\n",
                        ms > 0 ? 1000.0 * (double) produced / ms : 0.0, produced > 0 ? ms / (double) produced : 0.0,
                        st.resident, st.vram_gb, st.pinned_gb, (double) (st.staged - s_dec.staged) * per);
        };
        std::string fail;
        for (;;) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            if (in_think && (produced >= r.think_budget || in.wrap.load()) && tok != r.think_end) tok = r.think_end;
            std::printf("T %d\n", tok);
            ++produced;
            if (in_think && tok == r.think_end) in_think = false;
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            if (std::find(eos.begin(), eos.end(), tok) != eos.end()) {
                finish = "stop";
                break;
            }
            if (produced >= r.max_new || (int) state.size() + 1 >= m.max_ctx()) break;
            if (!m.forward(&tok, 1, (int) state.size(), 1, lg.data(), err)) {
                fail = err;
                break;
            }
            state.push_back(tok);
            tok = mm3::sample::draw(mm3::sample::dist(lg.data(), m.vocab(), r, state), rng);
        }
        if (!fail.empty()) {
            std::fprintf(stderr, "nh: %s\n", fail.c_str());
            std::printf("ERR %s\n", fail.c_str());
            finish = "cancel";
            state.clear();
        }
        stat_line();
        const double t2 = now() * 1e3;
        const nh::Stats& se = m.stats();
        const long long look = se.profiled - s_dec.profiled;
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu %lld %lld\n", produced, r.ids.size(), t1 - t0, t2 - t1, finish,
                    from, look, look);
        std::fflush(stdout);
    }
    return 0;
}

int usage() {
    std::fprintf(stderr, "usage: maya-nh --pack DIR --ids IDS|@FILE [--gen N] [--dump-logits FILE --last N] [options]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    nh::Options o;
    std::string ids_arg, dump, prof_out;
    int gen = 0, last = 1, split = 0;
    bool serve_mode = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--pack") o.pack = next();
        else if (a == "--ids") ids_arg = next();
        else if (a == "--gen") gen = std::atoi(next());
        else if (a == "--dump-logits") dump = next();
        else if (a == "--last") last = std::atoi(next());
        else if (a == "--gpu") o.gpu = std::atoi(next());
        else if (a == "--max-context") o.max_ctx = std::atoi(next());
        else if (a == "--chunk") o.chunk = std::atoi(next());
        else if (a == "--staging") o.staging = std::atoi(next());
        else if (a == "--reserve-gb") o.reserve_gb = std::atof(next());
        else if (a == "--resident-cap") o.resident_cap = std::atoi(next());
        else if (a == "--profile") o.profile = next();
        else if (a == "--profile-out") prof_out = next();
        else if (a == "--no-pin-pool") o.pin_pool = false;
        else if (a == "--split") split = std::atoi(next());
        else if (a == "--serve") serve_mode = true;
        else if (a == "-v" || a == "--verbose") o.verbose = true;
        else return usage();
    }
    if (o.pack.empty() || (ids_arg.empty() && !serve_mode)) return usage();
    std::vector<int> ids;
    if (!serve_mode && !read_ids(ids_arg, ids)) {
        std::fprintf(stderr, "maya-nh: no token ids in %s\n", ids_arg.c_str());
        return 2;
    }
    nh::Model m;
    std::string err;
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-nh: load: %s\n", err.c_str());
        if (serve_mode) {
            std::printf("ERR load: %s\n", err.c_str());
            std::fflush(stdout);
        }
        return 1;
    }
    if (serve_mode) return serve(m, o.chunk);
    for (const std::string& l : m.info_lines()) std::fprintf(stderr, "%s\n", l.c_str());
    const int V = m.vocab();
    last = std::max(1, std::min(last, (int) ids.size()));
    std::vector<float> lg((size_t) last * V);
    const double t0 = now();
    const int step = split > 0 ? split : (int) ids.size();
    for (int i = 0; i < (int) ids.size(); i += step) {
        const int n = std::min(step, (int) ids.size() - i);
        const int left = (int) ids.size() - i - n;
        const int want = std::max(0, std::min(n, last - left));
        if (!m.forward(ids.data() + i, n, i, want, want ? lg.data() + (size_t) (last - left - want) * V : nullptr, err)) {
            std::fprintf(stderr, "maya-nh: forward: %s\n", err.c_str());
            return 1;
        }
    }
    const double t1 = now();
    std::fprintf(stderr, "prompt: %zu tokens in %.2f s (%.1f tok/s), %ld experts staged\n", ids.size(), t1 - t0,
                 ids.size() / (t1 - t0), m.stats().staged);
    {
        const float* r = lg.data() + (size_t) (last - 1) * V;
        std::vector<int> ix((size_t) V);
        for (int i = 0; i < V; ++i) ix[(size_t) i] = i;
        std::partial_sort(ix.begin(), ix.begin() + 10, ix.end(), [&](int a, int b) { return r[a] > r[b]; });
        std::printf("top:");
        for (int i = 0; i < 10; ++i) std::printf(" %d:%.3f", ix[(size_t) i], r[ix[(size_t) i]]);
        std::printf("\n");
    }
    if (!dump.empty()) {
        std::FILE* f = std::fopen(dump.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "maya-nh: cannot write %s\n", dump.c_str());
            return 1;
        }
        const int hdr[4] = {last, V, (int) ids.size(), 0};
        std::fwrite("NHL1", 1, 4, f);
        std::fwrite(hdr, 4, 3, f);
        std::fwrite(ids.data(), 4, ids.size(), f);
        std::fwrite(lg.data(), 4, lg.size(), f);
        std::fclose(f);
    }
    if (gen > 0) {
        std::vector<float> one((size_t) V);
        int nxt = (int) (std::max_element(lg.end() - V, lg.end()) - (lg.end() - V));
        const double g0 = now();
        int done = 0;
        std::printf("gen:");
        for (int i = 0; i < gen; ++i) {
            std::printf(" %d", nxt);
            std::fflush(stdout);
            ++done;
            const std::vector<int> eos = m.eos();
            if (std::find(eos.begin(), eos.end(), nxt) != eos.end()) break;
            if (i + 1 == gen) break;
            if (!m.forward(&nxt, 1, m.pos(), 1, one.data(), err)) {
                std::fprintf(stderr, "\nmaya-nh: decode: %s\n", err.c_str());
                return 1;
            }
            nxt = (int) (std::max_element(one.begin(), one.end()) - one.begin());
        }
        const double g1 = now();
        std::printf("\n");
        std::fprintf(stderr, "decode: %d tokens in %.2f s (%.2f tok/s), %ld experts staged\n", done - 1, g1 - g0,
                     (done - 1) / std::max(1e-9, g1 - g0), m.stats().staged);
    }
    if (!prof_out.empty() && !m.save_profile(prof_out, err)) {
        std::fprintf(stderr, "maya-nh: %s\n", err.c_str());
        return 1;
    }
    return 0;
}
