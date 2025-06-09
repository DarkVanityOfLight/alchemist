# Alchemist

Alchemist is a transpiler that is used to convert files of the [Armoise](https://tapas.labri.fr/wp/?page_id=17) language 
into SMT2-Lib.


## Main Usage
The transpiler was designed to translate the output of the [FASTer](https://tapas.labri.fr/wp/?page_id=23) tool into SMT2 SMT2-Lib.
This can be used to automatically proof termination of a program modeled using FASTer, by checking for an
infinite ramsey clique in the relation using [my](https://github.com/DarkVanityOfLight/RamseyLinearArithmetics) tool.
