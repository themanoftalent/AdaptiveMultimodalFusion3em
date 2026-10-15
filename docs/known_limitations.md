# Known limitations

1. The TCIA histopathology development data include only four patients.
2. The public radiograph and histopathology datasets are unpaired.
3. The radiograph-to-histopathology probability remapping is heuristic.
4. Mapping `P_other` to `Non tumor` creates a class-definition mismatch because the radiograph `Other` class represents non-osteosarcoma bone tumors rather than histologically normal tissue.
5. Splitting `P_tumor` equally into viable and necrotic tumor probabilities structurally increases entropy at the viable-versus-necrotic boundary.
6. The Hitit cohort cannot be publicly released because of ethics approval and institutional data-governance restrictions.
7. The repository reproduces the computational workflow, but exact numeric reproduction requires the same dataset versions, split files, preprocessing settings, and software environment.
