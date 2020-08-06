class mnli_data:
    def __init__(self, all_labels):
        self.all_labels = all_labels

    def convert_to_mnli_format(self, id, text, label=None):
        if label:
            ids = [id]
            texts_1 = [text]
            texts_2 = ["this text is {}".format(label)]
            labels = ["entailment"]
            orig_label = [label]
        else:
            ids = []
            texts_1 = []
            texts_2 = []
            labels = []
            orig_label = []

        for l in self.all_labels:
            if label:
                if l != label:
                    ids.append(id)
                    texts_1.append(text)
                    texts_2.append("this text is {}".format(l))
                    labels.append("contradiction")
                    orig_label.append(l)
            else:
                ids.append(id)
                texts_1.append(text)
                texts_2.append("this text is {}".format(l))
                orig_label.append(l)

        return ids, texts_1, texts_2, labels, orig_label