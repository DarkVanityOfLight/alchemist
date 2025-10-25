# Alchemist

**Alchemist** is a transpiler that converts files written in the [Armoise](https://tapas.labri.fr/wp/?page_id=17) language into **SMT-LIB v2** format. It is designed to translate outputs from the [FASTer](https://tapas.labri.fr/wp/?page_id=23) tool, enabling automated verification of program termination using Ramsey-based techniques via [this tool](https://github.com/DarkVanityOfLight/RamseyLinearArithmetics).

## Features

* Converts Armoise specifications to SMT-LIB v2.
* Works seamlessly with FASTer outputs.
* Supports termination proofs using infinite Ramsey clique detection.
* Command-line interface with flexible options for output, verbosity, and timing.

## Installation

Alchemist uses Python and Pipenv for dependency management:

```bash
git clone https://github.com/DarkVanityOfLight/alchemist
cd alchemist
pipenv install
```

## Usage

```bash
python alchemist.py [options] <files>
```

**Positional Arguments:**

* `files` – One or more.ams source files to compile.

**Options:**

* `-r, --relation-name` – Set the output relation name (default: `R`).
* `-o, --output` – Write output to a file (defaults to STDOUT).
* `-v, --verbose` – Enable verbose logging.
* `-t, --time` – Measure and report total compilation time.

**Examples:**

```bash
# Compile one or more.ams files
python alchemist.py file1.ams file2.ams

# Set a custom relation name
python alchemist.py input.ams -r MyRelation

# Save output to a file
python alchemist.py input.ams -o output.smt2

# Enable verbose logging
python alchemist.py input.ams -v

# Measure compilation time
python alchemist.py input.ams -t
```

## Architecture overview
Alchemist is a compiler-like tool chain for the Armoise language,
designed to parse mathematical set and predicate expressions and emit
SMT-LIB2 code for formal reasoning and verification. The architecture
consists of several modules, each responsible for a distinct phase of
processing, from lexical analysis to IR optimization and SMT emission.

### Frontend

The frontend covers **lexing**, **parsing** and **AST construction**. We
tokenize Armoise source using PLY(Python Lex-Yacc) handeling
operators, identifiers, numbers, and comments while tracking source
locations for error reporting. From the lexed tokens we use PLY’s Yacc
to build an abstract syntax tree(AST), representing constructs such as
set operations, comprehensions, and predicates. The AST is **sibling
tree**, meaning each node has one (optional) child and a list of
siblings.

### Intermediate Representation

The AST is converted into an intermediate representation (IR) that
encodes the mathematical semantics of Armoise. During this process,
scoped let ... in bindings are resolved and globalized, so that every
identifier is replaced by a unique global ID.

The IR is organized as a collection of rooted trees, each representing
an expression or definition. To support sharing and reuse, sub-trees can
reference other trees by their global IDs. This means the IR is best
understood as a **forest of trees with cross-references**, which
together form a directed acyclic graph (DAG). Each IR node stores an ID,
type information, and children, while references are used to resolve
local definitions.

The DAG property is assumed but not strictly enforced and cycles are not
checked for explicitly.

### Optimization and IR transformations

The optimizer applies two main transformations to simplify and normalize
the IR before SMT emission: identifier inlining and linear-transform
pushdown. Both operate under the assumption that the IR forms a DAG
(acyclic, but with shared subtrees via global identifiers).

#### Identifier inlining

Inlining removes the references by expanding identifiers into their
definitions. The process has two phases:

1.  **Dependency analysis.** The IR is traversed to collect identifier
    definitions and build a dependency graph between them. A topological
    sort determines a valid expansion order. Cycles are not expected. If
    present, they are logged but not resolved.

2.  **Substitution.** Following the order, each identifier is replaced
    with its fully inlined definition. Substitutions are memoized,
    ensuring consistency and preserving shared structure. Finally,
    identifiers in the root expression are replaced, leaving a
    definition-free IR.

This guarantees that all identifiers refer directly to their underlying
definitions, reducing lookups and enabling later structural rewrites.
Note that the inlining process can impose an exponential blowup.

#### Linear-transform pushdown

Linear transforms (scales and shifts applied to arguments) are
normalized by pushing them as deep into the IR as possible. A
LinearTransform wrapper records scale/shift factors and composes them as
the traversal continues.

The algorithm:

- **Absorbs** nested *LinearScale* and *Shift* nodes by folding their
  factors into the current transform, recording scale metadata in
  annotations when relevant.

- **Distributes** transforms over unions, intersections, and
  union-spaces by applying the transform to each part independently.

- **Pushes** into complements, comprehensions, and guards by wrapping
  sub-expressions with identity transforms, keeping per-argument factors
  explicit but localized.

- **Applies** the arithmetic directly to vector literals.

- Unsupported node types raise errors explicitly, preventing silent
  mis-rewrites.

Together, these optimizations yield a normalized, definition-free IR
that is easier to translate into efficient SMT-LIB2 constraints.

### Backend

The backend translates the optimized IR of an Armoise predicate into
SMT-LIB2 function definitions suitable for SMT solvers. It handles set
comprehensions, vector spaces, linear transformations, arithmetic and
logical guards, and domain constraints.

The emission process is recursive, mapping each IR construct to its
corresponding SMT-LIB2 representation. Compound constructs (unions,
intersections, complements) are translated into logical connectives, and
arithmetic expressions are flattened and normalized for solver
compatibility. Linear transformations are encoded as scaled and shifted
variable relations, including modular constraints where present, while
set comprehensions combine domain constraints with guard conditions.

Guards and arithmetic constraints are emitted in a way that preserves
the semantics of the IR, ensuring correct mapping of variables to SMT
function parameters and enforcing type-specific domain constraints.

The result is a fully recursive, normalized SMT-LIB2 function faithful
to the original Armoise predicate, ready for use with SMT solvers.

## References

* [Armoise Language](https://tapas.labri.fr/wp/?page_id=17)
* [FASTer Tool](https://tapas.labri.fr/wp/?page_id=23)
* [Ramsey Linear Arithmetics](https://github.com/DarkVanityOfLight/RamseyLinearArithmetics)
