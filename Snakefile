from pathlib import Path
import pandas as pd


MFA_IMAGE = "/home/jgauthier/mfa.sif"

corpus_id = "33062e87"
AUDIO_STEMS = [os.path.splitext(f)[0] for f in Path(f"materials/{corpus_id}/tts").glob("B*/*.wav")]


rule output_transcript:
    input:
        csv = "materials/{corpus_id}/{corpus_id}-sentences-manual.csv",
        audio = "materials/{corpus_id}/tts/B{block}/B{block}_{word}_{label}_{corpus_id}.wav"

    output:
        txt = "materials/{corpus_id}/tts/B{block}/B{block}_{word}_{label}_{corpus_id}.txt",

    run:
        block_number = int(wildcards.block)

        # first prepare the single-line text file
        df = pd.read_csv(input.csv, index_col=0).loc[block_number]
        df = df.loc[(df.target_word == wildcards.word) & (df.label == wildcards.label)]
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"

        with open(output.txt, "w") as f:
            f.write(df.sentence.iloc[0])


# Download acoustic model + dictionary for mfa
rule prepare_mfa:
    output:
        acoustic = "mfa/pretrained_models/acoustic/english_us_arpa.zip",
        dictionary = "mfa/pretrained_models/dictionary/english_us_arpa.dict"

    shell:
        """
        mkdir mfa
        singularity exec -B "`pwd`" -B "`pwd`/mfa:/mfa" {MFA_IMAGE} mfa download acoustic english_us_arpa
        singularity exec -B "`pwd`" -B "`pwd`/mfa:/mfa" {MFA_IMAGE} mfa download dictionary english_us_arpa
        """


rule align_audio:
    input:
        corpus = f"materials/{corpus_id}/tts",
        corpus_audio = expand("{stem}.wav", stem=AUDIO_STEMS),
        corpus_txt = expand("{stem}.txt", stem=AUDIO_STEMS),
        acoustic = "mfa/pretrained_models/acoustic/english_us_arpa.zip",
        dictionary = "mfa/pretrained_models/dictionary/english_us_arpa.dict"

    output:
        expand("{stem}.TextGrid", stem=AUDIO_STEMS)

    shell:
        """
        # remove any cached data from previous MFA run
        rm -rf mfa/tts

        singularity exec -B "`pwd`" \
            -B "`pwd`/mfa:/mfa" \
            {MFA_IMAGE} mfa align \
            "{input.corpus}" \
            english_us_arpa english_us_arpa \
            "{input.corpus}"
        """


rule align_audio_all:
    input:
        expand("{stem}.TextGrid", stem=AUDIO_STEMS)
    output:
        "materials/{corpus_id}/tts.done"
    shell:
        "touch {output}"