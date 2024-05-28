def get_fastqs(wildcards):
    d = {}

    if layouts[wildcards.sample] == "single":
        d["r1"] = f"/home/mccuem/data_release/umgc/nextseq/180108_NB551164_0013_AHLGWKBGX3/McCue_Project_033/{wildcards.sample}_R1_001.fastq.gz"
    elif layouts[wildcards.sample] == "paired":
        d["r1"] = f"/panfs/roc/groups/3/mccuem/dimml002/NuGEN/FASTQs/{wildcards.sample}_R1_001.fastq.gz",
        d["r2"] = f"/panfs/roc/groups/3/mccuem/dimml002/NuGEN/FASTQs/{wildcards.sample}_R2_001.fastq.gz"
    
    return d
