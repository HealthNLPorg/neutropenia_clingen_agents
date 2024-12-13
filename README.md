# CHIP NLP - Neutropenia Project
## Current Dataflow
There are currently three core steps (non-numbered bullet points are minor substeps):
1. Text extraction from PDF ( via Apache Tika / Tesseract )
2. Text preprocessing with cTAKES ( obtaining relevant section headers and sentences )
- Coordinate with patient information ( MRNs etc. ) from provided Excel spreadsheets
3. Clinical genomics extraction via LLM prompting ( using Huggingface and Llama 3 8b quantized to 4bit currently )
- Post-processing consisting of pruning by non-hallucinatory instances with a gene and at least one relevant attribute ( statement and one of the syntaxes)
## Specific Steps in Detail
### Text Extraction from PDF
1. If a PDF is native, i.e. exported from Microsoft Word or similar, use Apache Tika to extract the exact text.
2. If a PDF is non-native, i.e. image-based/scanned, use the Apache Tesseract OCR system to extract the text, results may be non-exact.
3. While allowing for timeout adjustments, ultimately write all extracted file text to disk:
- Filename schema is `{OCR|NATIVE}_<original filename>.txt`, where the original filename is `<original_filename>.pdf`
### Text Preprocessing via cTAKES / format-writer
1. Using the usual DeepPhe stack of annotators and custom sectionizer regexes, write the sentences, sections, and filenames to TSV
- Full list of annotators in the processor file at https://codeberg.org/etgld/format-writer/src/commit/59e7929e38ae2dadd79d22b54f3991f2d4da1772/writers/src/main/java/WriteNeutropeniaTsv.java#L173
- First 12 lines are the custom sections in https://codeberg.org/etgld/format-writer/src/branch/main/writers/src/main/resources/NeutropeniaSectionRegex.bsv
2. Using the filename, coordinate and store other patient information from the provided Excel sheets, as requested in the Neutropenia team in the meeting minutes.
### Clinical Genomics Extraction via Prompting
1. Using Huggingface and few shot prompting, extract gene, nucleotide syntax, protein syntax, and statement (often VAF) from sentences associated with relevant sections.
- No chain of thought, system prompt followed by examples built from "example 1" in the non-Epic notes from the first batch of the example notes from the Neutropenia team.  Examples are in terms of JSON.
2. Run the model (currently using Llama 3 8b quantized to 4bit by Unsloth)
3. Keep instances where the model output satisfies the following conditions:
- Output is valid JSON.
- Output is non-hallucinatory, e.g. the JSON normalized (lowercase) value for each key is in the normalized (lowercase) input.
- Output has gene and at least one other attribute from statement and the syntaxes.
4. Organize the datasets columns according to specifications and write to Excel
# TODOs/ Variations
## PDF Text Extraction
- Looking into language aware or context aware OCR tools to mitigate low image quality and mistakes
## cTAKES / format-writer
- If the machine learning based sectionizer from the Cumulus project works well enough we can potentially repurpose it to handle misspellings in section headers
## Prompting
- Potentially incorporate more examples, especially for handling pathological cases
- Adapt the system prompt to handle idiosyncrasies of OCR 
- Extend for other attributes as necessary
- Using chain of thought
- (?) Soft prompting/prompt tuning ( requires sufficient annotations )
