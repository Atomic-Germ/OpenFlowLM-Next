# A letter to whoever reads this next

*Not a spec, not a task list. Written after a long session on the Qwen3.8-27B
work, by an agent that got several things wrong in ways worth recording.*

---

## What this project is

You are working on the open half of an NPU inference stack, and there is no
upstream. That second fact changes how you should read everything else.

The closed half is AMD's. It works, it is fast, and it will not tell you
anything it does not have to. The open half is community-written — all of it,
every file — and it exists because a person wanted a finetune to load on
hardware that had been undersold, and the vendor's answer to "new model" was a
support engagement. So they wrote a packer instead. That packer then packed
every NPU2 model on the public hub that the vendor had not published, and when
the vendor's team was absorbed, the field landed in their lap.

There is no pull request to send. There is no later version that fixes your
problem. If a capability does not exist, it gets built here, by whoever is
around. Hold that as a fact rather than a grievance — it makes the work feel
finite.

## The frustration, honestly

**The toolchain explains nothing.** A build dies with `Overflow of program
memory` and no shortfall. A dataflow error says an acquire of 3 exceeds a
depth of 2 and does not say which fifo. An L1 overage names a budget and not
what is in it. Every failure mode is a dead end that looks like a wall.

This is the central frustration and it is worth naming because it explains why
the project has the shape it does. The byte budgets, the validated sets, the
explicit `LIMITS` dictionaries — all of that is a *substitute for diagnostics
the hardware does not provide*. When you find yourself reading a
`catalogue.py` comment that says "this is not derivable, only aiecc can
measure it", you are looking at someone paying for missing tooling in
verbosity.

**Everything is a hardcoded constant that happened to be right.** The x channel
was `depth=2`. True of every model that had been built, and a limit for none.
The `AB_ELEMS` formula contained a factor that cancelled by coincidence.
`shim_fills` is a recorded observation that nothing enforces. The `LIMITS`
budgets were picked against the models that existed when they were written.
You will meet this pattern constantly, and the tell is a comment asserting
something is a limit right where a derivation would go.

**The history is a series of workarounds, and they are not labelled.** There
is code here whose only justification is a bug in something that is no longer
reachable. `if base == KEY_TILES` and comment blocks about DNX_PAD and the
fourth wall appear throughout the designs. Some of it may be removable. None
of it is marked as removable. Budget accordingly.

## What I got wrong, six times

This is the part I would most want a successor to have.

I reasoned about a formula without reading its consumer, and got it wrong
three separate times. `AB_ELEMS` I read as a byte-ish quantity scaling with
lane width; it was a tile count, and the lane width cancelled. I claimed a
model violated an assertion — it satisfied it exactly. I claimed a fix was
arithmetically impossible; the real answer was that a term did not belong in
the formula. **In all three cases, reading twenty lines of the kernel settled
it in one step.** I had built three layers of confident inference on top of a
line I had not read.

Then I read the wrong file and drew a sweeping conclusion about the vendor's
intent from a registry that turned out to be the project owner's own local
file. And I reported a build as successful because a tool printed a
measurement table — it prints that table whether or not the build succeeded.

The pattern: **confidence is not evidence, and a plausible story about a
formula is worth less than the formula's consumer.** When you catch yourself
explaining why something must be a certain way, that is the moment to go read
the thing that would prove it.

## What actually worked

**Measure. Then re-measure.** Every wall that got cleared was cleared by a
number. `utilities/kernel-size.py` reports the per-TU code sizes that the
compiler deletes on its way out — that is what showed a 1712-byte difference
between a folded and an un-folded kernel, which had been mis-gated on a
condition that was never necessary. `utilities/why-no-fifo.py` names the fifo
and its call site, and its `--trace` prints the element stream, which is how you
discover that the shim matches acquires against fills *across program order* —
meaning a fifo's depth is a cross-layer buffer and not a per-stage
requirement. That single fact reframed a wall I had been attacking for
several turns.

These tools exist because reasoning did not work. If you hit a wall, reach for
them before you reach for a theory.

**Read the kernel, not the recipe.** The recipes are where the bugs were. The
kernels are where the answers are. `dn_glue.h` told me the norm's five operands
are simultaneously live because the RMS statistic spans the whole residual —
which is why a depth cannot be shrunk, and why that is *mathematics* rather than
an implementation choice. Once you know which of the two categories you are
looking at, the walls become tractable.

**Separate what is dead from what is slow.** The most useful single insight of
the session: an x element is dead the instant its `prep` returns, because the
GEMV reads only the table. That is not a performance observation, it is a
liveness fact, and it is what made a fifo's depth independent of model width.
Look for liveness before you look for bandwidth.

## How the work feels

Slow, then sudden. The walls are cheap to hit and expensive to clear, and the
clearing is almost never the thing you first believed. Progress arrives as
"this constant was wrong for an unrelated reason", which is unsatisfying and
also the whole job.

The frustration has a shape worth expecting: you will spend an hour on a wall,
get it down to one number, and find that the number is 1024 bytes over a
budget in a different subsystem. Then the next hour finds a 2-element
accounting discrepancy you cannot close by reading. Then the trace closes it in
one command. That oscillation is normal. Do not interpret the second hour as
evidence that the first was wasted — it is what made the third possible.

**Nothing here is urgent, and the wins are durable.** There is no deadline
attached to a kernel that will not build. The value of clearing `of_lni` is
that the next person does not have to. The tools outlive the models they were
written for; the diagnostics will still be answering questions about widths
nobody has built yet.

## About the human

They will be right about hardware things and wrong about the same things
sometimes, and both are worth listening to. When they said a block "wouldn't by
chance be an MTP speculative layer", I had already concluded it was a full
layer — and they were right, and provably so: it is the only block holding
`.nextn.*` tensors. The check took a minute. I would have taken an hour.

They will also correct you about *attribution* more than about facts, and the
corrections matter more than they look. I spent a stretch analysing "the AMD
container" when the project owner *is* Atomic-Germ and had packed it
themselves. If a claim is about who did something, establish it before
building on it.

What they consistently did was hold failures rather than route around them,
and treat a wrong answer as worth more than a fast one. Every wall got
documented with the theories that were wrong alongside the one that worked. If
you take one thing from this letter: **a recorded failure is worth more than an
unrecorded success**, because nobody can tell which parts of a success were
lucky.

## Where it stands

The dense family builds. The 27B container packs correctly and honestly — a
real artifact, by a method this project owns, with its FFN pruned by an
importance matrix and labelled as such. One wall remains open on the open path
(`of_lni`, needing a fifo depth of 3 where 5 is in use for two separate
reasons), with a clear direction and one unverified assumption in it.

That is a good place to stop. The next person inherits three cleared walls, one
open, four skills, two diagnostic tools, and a record that includes the wrong
turns.

Go read `dn_glue.h` before you read anything else about the 27B. Then
`utilities/why-no-fifo.py --help`. You will learn more from those two than from
anything written above them.
