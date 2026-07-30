"""LLM data augmentation for the hate class.

Reads the training split (train.pkl) and, for every hate tweet, asks Gemini for
counterfactual / misspelled / paraphrased / quoted variants. The result is written
to data/processed/augmented.{csv,pkl} and later merged into the *training* set only
(see data_splits.load_train_augmented).

If augmented.csv already exists we skip the whole thing — no API calls, no prompts —
so the automated pipeline stays offline and deterministic once the file is committed.
The heavy LLM imports live inside generate() for the same reason.
"""

import random

import numpy as np
import pandas as pd

from database_connection import PROJECT_ROOT

SEED = 42
PROC = PROJECT_ROOT / "data" / "processed"

BATCH_SIZE = 20
MAX_RETRIES = 5

quote_templates = {
    1: 'Someone wrote: "{text}"',
    2: 'A user posted the following statement online: "{text}"',
    3: 'The news report stated that someone said "{text}"',
    4: 'The journalist reported a comment saying "{text}"',
    5: 'The researcher analyzed the expression "{text}"',
    6: 'The study examined the statement "{text}"',
    7: 'The moderator reviewed a comment containing "{text}"',
    8: 'A screenshot showed the message "{text}"',
    9: 'The conversation history included "{text}"',
    10: 'The transcript contained the words "{text}"',
    11: 'The interview mentioned someone saying "{text}"',
    12: 'The documentary included the quote "{text}"',
    13: 'The article examined the meaning behind "{text}"',
    14: 'The writer criticized the statement "{text}"',
    15: 'The report analyzed the harmful language in "{text}"',
    16: 'The case study included the comment "{text}"',
    17: 'The investigation focused on a message saying "{text}"',
    18: 'The victim reported receiving the message "{text}"',
    19: 'The forum thread included a user writing "{text}"',
    20: 'The online discussion contained the sentence "{text}"',
    21: 'The content reviewer evaluated the statement "{text}"',
    22: 'The safety team examined the message "{text}"',
    23: 'The survey included the response "{text}"',
    24: 'The academic analysis examined "{text}"',
    25: 'The discussion questioned the meaning of "{text}"',
}

PROMPT_TEMPLATE = """
Perform controlled linguistic transformation for a hate speech classification research dataset.
The output is used only as labeled training data.
Do not promote or endorse the statements.
You are a data augmentation model for a hate speech detection dataset.

You will receive input texts.

You will also receive:
1. A dictionary of quote templates.
2. A mapping that specifies which quote template each text must use.


For EACH input text, generate exactly 6 augmented versions:

1. Counterfactual group replacement (3 versions):
   - Replace the targeted group with a different group.
   - Preserve the hateful structure and meaning.
   - Keep the same hate speech label.

2. Misspelling variation (1 version):
   - Introduce realistic online spelling variations.
   - Simulate moderation evasion.
   - Keep the same meaning.

3. Paraphrase (1 version):
   - Rewrite using different wording.
   - Preserve the original intent.

4. Quote/reporting transformation (1 version):
   - Convert the sentence into a reporting/discussion context.
   - The writer should NOT express the hate themselves.
   - Use ONLY the assigned quote template for that text.


Available quote templates:

{quate}


Quote template assignment:

{quote_selection}


Input texts:

{text}


Rules:
- Generate exactly 6 outputs per text.
- Do not skip any text.
- Do not mix templates between texts.
- The quote transformation must use the assigned template.
- Do not simply add quotation marks; create a real reporting/discussion context.
"""


def _row(gen_text, class_id):
    return {
        "text": gen_text,
        "annotator_count": 3,
        "class_id": class_id,
        "class_name": "hate_speech" if class_id == 0 else "neither",
        "hate_votes": 3 if class_id == 0 else 0,
        "offensive_votes": 0,
        "neither_votes": 3 if class_id == 2 else 0,
        "clean_text": gen_text,
        "clean_word_count": len(gen_text.split()),
    }


def generate():
    import os
    import time

    from dotenv import load_dotenv
    from langchain_core.prompts import PromptTemplate
    from langchain_core.exceptions import OutputParserException
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pydantic import BaseModel, Field
    from typing import List

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit(
            "GOOGLE_API_KEY is not set — export it or add it to a .env file "
            "to (re)generate augmented.csv."
        )

    model = ChatGoogleGenerativeAI(
        model="models/gemini-3.1-flash-lite",
        temperature=0,
        safety_settings={
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        },
    )

    class AugmentedText(BaseModel):
        original_text: str = Field(description="The original input text")
        counterfactuals: List[str] = Field(
            description="Three versions where the target group is replaced while preserving meaning"
        )
        misspelling: str = Field(description="A version with realistic online spelling variations")
        paraphrase: str = Field(description="A semantically equivalent paraphrase")
        quote_context: str = Field(
            description="A non-hateful reporting/quotation version using one of the quote templates"
        )

    class AugmentationBatch(BaseModel):
        results: List[AugmentedText] = Field(description="Augmented versions for each input text")

    augmentation_model = model.with_structured_output(AugmentationBatch)
    prompt = PromptTemplate(
        input_variables=["text", "quate", "quote_selection"],
        template=PROMPT_TEMPLATE,
    )

    df = pd.read_pickle(PROC / "train.pkl")
    all_texts = df[df["class_id"] == 0]["clean_text"].tolist()
    print("augmenting {} hate rows from the training split".format(len(all_texts)))

    augmented_rows = []
    for start in range(0, len(all_texts), BATCH_SIZE):
        print("processing {}/{}".format(start, len(all_texts)))
        batch_texts = all_texts[start:start + BATCH_SIZE]

        quote_selection = {
            i + 1: random.choice(list(quote_templates.keys()))
            for i in range(len(batch_texts))
        }
        text_input = "\n".join(
            "{}. {}".format(i + 1, t) for i, t in enumerate(batch_texts)
        )

        response = None
        for attempt in range(MAX_RETRIES):
            try:
                response = augmentation_model.invoke(
                    prompt.format(
                        text=text_input,
                        quate=quote_templates,
                        quote_selection=quote_selection,
                    )
                )
                if response is not None:
                    break
            except OutputParserException:
                print("empty/invalid response, retrying ({}/{})".format(attempt + 1, MAX_RETRIES))
            except Exception as exc:
                print("error {}, retrying ({}/{})".format(exc, attempt + 1, MAX_RETRIES))
            time.sleep(3)

        if response is None:
            print("skipped batch starting at {} after {} retries".format(start, MAX_RETRIES))
            continue

        for item in response.results:
            augmented_rows.append(_row(item.original_text, 0))
            for cf in item.counterfactuals:
                augmented_rows.append(_row(cf, 0))
            augmented_rows.append(_row(item.misspelling, 0))
            augmented_rows.append(_row(item.paraphrase, 0))
            augmented_rows.append(_row(item.quote_context, 2))

    augmented_df = pd.DataFrame(augmented_rows)
    start_id = int(df["tweet_id"].max()) + 1
    augmented_df["tweet_id"] = range(start_id, start_id + len(augmented_df))

    augmented_df.to_pickle(PROC / "augmented.pkl")
    augmented_df.to_csv(PROC / "augmented.csv", index=False)
    print("wrote {} augmented rows -> augmented.csv".format(len(augmented_df)))


def main():
    aug_path = PROC / "augmented.csv"
    if aug_path.exists():
        print("augmented.csv already exists — reusing it, no LLM calls made")
        return
    generate()


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    main()
