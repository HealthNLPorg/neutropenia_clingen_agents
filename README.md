# CHIP NLP - Neutropenia Project
## Current Dataflow
There are currently three core steps (non-numbered bullet points are minor substeps):
1. Text extraction from PDF ( via Apache Tika / Tesseract )
2. Text preprocessing with cTAKES ( obtaining relevant section headers and sentences )
- Coordinate with patient information ( MRNs etc. ) from provided Excel spreadsheets
3. Clinical genomics extraction via LLM prompting ( using Huggingface and Llama 3 8b quantized to 4bit currently )
- Post-processing consisting of pruning by non-hallucinatory instances with a gene and at least one relevant attribute ( statement and one of the syntaxes)

# TODOs
Reminder, migrate the Tika processing code from Cumulus Doc Type repo to figure things out back here
