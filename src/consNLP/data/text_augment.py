def augment(text,label,attack,num_augments=3):
    '''
    e.g.
    import textattack
    import nlpaug

    textattack.augmentation.WordNetAugmenter()
    textattack.augmentation.EmbeddingAugmenter()
    nlpaug.augmenter.word.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
    nlpaug.augmenter.sentence.ContextualWordEmbsForSentenceAug(model_path='gpt2')
    nlpaug.augmenter.word.SynonymAug(aug_src='wordnet', lang='eng')

    '''

    if type(label) == list:
        print ("Augmentation does not support sequence output")
        return text, label
    else:
        augmented_texts = attack.augment(text)
        labels = [label]*len(augmented_texts)

        return augmented_texts, labels
