import logging
from typing import Tuple, List, Union
import pandas
import spacy
from simpletransformers.ner import NERModel
from spacy.tokens.doc import Doc


logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("simpletransformers").setLevel(logging.ERROR)


class TextLabeler:

    def __init__(self):
        print("Loading DeBERTaV3 model --> ", end="")
        self.model = NERModel("deberta-v2", "trained_models/microsoft_deberta-v3-base/")
        print("done")

        print("Loading SpaCy model en_core_web_lg --> ", end="")
        self.nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "textcat"])
        self.nlp.add_pipe('sentencizer')
        print("done")

    def label(self, unlabeled_text: str) -> Union[List[List[Tuple[str, str]]], None]:
        """
        Label a review by first preprocessing it using the loaded SpaCy model, converting it into simpletransformers-
        readable format and then running simpletransformers model inference on it.

        :param unlabeled_text: review text to label
        :return: string tuple list nested by sentences where each tuple contains token, prediction
        """
        processed_df = self.get_processed_df(self.nlp(unlabeled_text))
        if processed_df is not None:
            tokens, predicted_labels_sentiment = processed_df
            labeled_output = []
            for toks, preds in zip(tokens, predicted_labels_sentiment):
                labeled_output.append((list((t, p) for t, p in zip(toks, preds))))
            return labeled_output
        return None

    def get_processed_df(self, processed_doc: spacy.tokens.doc.Doc) -> Union[Tuple[List[List[str]], List[List[str]]], None]:
        """
        Helper method to convert processed SpaCy doc object containing word and sentence tokenization information into
        simpletransformers-format pandas DataFrame, run model inference on it and return nested lists of tokens and
        predicted labels.

        :param processed_doc: processed SpaCy doc object
        :return: tuple (tokens, labels) where both are string lists nested by sentence
        """
        sent_starts = [i for i, token in enumerate(processed_doc) if token.is_sent_start]
        sent_spans = []
        for i, s in enumerate(sent_starts):
            if i + 1 < len(sent_starts):
                sent_spans.append((s, sent_starts[i + 1]))
            else:
                sent_spans.append((s, len(processed_doc)))
        sentence_ids = []
        tokens = []
        nested_tokens = []
        labels = []
        for sent_id, sent_span in enumerate(sent_spans, start=1):
            sentence_ids += [sent_id for i, token in enumerate(processed_doc) if i in range(*sent_span)]
            tokens += [token.text for i, token in enumerate(processed_doc) if i in range(*sent_span)]
            nested_tokens.append([token.text for i, token in enumerate(processed_doc) if i in range(*sent_span)])
            labels += ["O" for i, token in enumerate(processed_doc) if i in range(*sent_span)]
            sent_id += 1
        df = pandas.DataFrame([(s_id, tok, lab) for s_id, tok, lab in zip(sentence_ids, tokens, labels)],
                              columns=["sentence_id", "words", "labels"])
        try:
            _, _, nested_labels = self.model.eval_model(df, verbose=False, silent=True)
        except ValueError:
            return None
        return nested_tokens, nested_labels
