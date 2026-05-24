import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='../data/processed/mixed_skypile_lccc.txt',
    model_prefix='my_tokenizer',
    vocab_size=32000,
    model_type='bpe',
    character_coverage=0.9998,
    num_threads=16,
    train_extremely_large_corpus=True,
    max_sentence_length=8192,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    pad_piece='[PAD]', unk_piece='[UNK]',
    bos_piece='[BOS]', eos_piece='[EOS]'
)

print("分词器训练完成！")