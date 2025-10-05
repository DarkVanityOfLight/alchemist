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
pipenv shell
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

## References

* [Armoise Language](https://tapas.labri.fr/wp/?page_id=17)
* [FASTer Tool](https://tapas.labri.fr/wp/?page_id=23)
* [Ramsey Linear Arithmetics](https://github.com/DarkVanityOfLight/RamseyLinearArithmetics)
