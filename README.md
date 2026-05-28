# Neutropenia Clinical Genomics Agents

Takes (cTAKES) sentences, gives them back with up to one genomic variant mention cluster of gene, protein and nucleotide change syntaxes, variant type (alternately assessed from section header),

# External Dependencies
Installing Tesseract OCR needs `sudo` unfortunately, best bet if you need it on someone else's server or HPC cluster is to install it within Apptainer.  Tika *should* work out of the box with the Python client but that has a history of not occuring. So you might need to download a `jar` file and do `java -jar ...tika..jar &` or something.
