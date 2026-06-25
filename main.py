import argparse

from funcs import (
    FIGURATIVE_PRESETS,
    scrape,
    scrape_edtekla,
    eda,
    annotate,
    split_sentences,
    fine_tune,
    figurative_train,
    predict_figurative,
    figurative_distill,
    figurative_eval_idioms,
)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bloomfield Plains Cree FNLP pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py --scrape
  python main.py --annotate
  python main.py --split-sentences --output data/sentences.txt --confidence 0.6
  python main.py --fine-tune --confidence 0.6
  python main.py --eda
""",
    )

    # data processing
    parser.add_argument("--scrape",          action="store_true",
                        help="Scrape Bloomfield texts to data/bloomfield_texts.csv")
    parser.add_argument("--scrape-edtekla",  action="store_true",
                        help="Download EdTeKLA parallel corpus and write src ||| tgt pairs")
    parser.add_argument("--edtekla-output",  default="data/sentences_edtekla.txt",
                        help="Output path for --scrape-edtekla (default: data/sentences_edtekla.txt)")
    parser.add_argument("--edtekla-append",  action="store_true",
                        help="Append EdTeKLA pairs to an existing file instead of overwriting")
    parser.add_argument("--eda",             action="store_true",
                        help="Run EDA and save figures to figures/")
    parser.add_argument("--annotate",        action="store_true",
                        help="Run LLM annotation on the Bloomfield paragraphs")
    parser.add_argument("--split-sentences", action="store_true",
                        help="Split paragraphs into sentence pairs and write src ||| tgt")
    parser.add_argument("--output",          default="data/sentences.txt",
                        help="Output path for --split-sentences (default: data/sentences.txt)")
    parser.add_argument("--confidence",      type=float, default=0.0,
                        help="Minimum alignment confidence filter for sentence pairs (0–1)")

    # tlm
    parser.add_argument("--fine-tune",       action="store_true",
                        help="TLM fine-tune a model on Cree-English sentence pairs")
    parser.add_argument("--sentences-file",  default=None,
                        help="Pre-split sentences file (cree ||| en) to use instead of the scraper pipeline")
    parser.add_argument("--model-name",      default="xlm-roberta-base",
                        help="Base model for --fine-tune (default: xlm-roberta-base)")
    parser.add_argument("--tlm-output-dir",  default="data/tlm_model",
                        help="Output directory for --fine-tune (default: data/tlm_model)")
    parser.add_argument("--epochs",          type=int,   default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size",      type=int,   default=16,
                        help="Per-device train batch size (default: 16)")
    parser.add_argument("--grad-accum",      type=int,   default=2,
                        help="Gradient accumulation steps for --fine-tune (default: 2)")
    parser.add_argument("--max-length",      type=int,   default=256,
                        help="Max token length per TLM input pair (default: 256)")

    # training args
    parser.add_argument("--hub-model-id",    default=None,
                        help="HuggingFace Hub repo ID to push the final model to")
    parser.add_argument("--wandb-project",   default=None,
                        help="Weights & Biases project name for logging (omit to disable)")
    parser.add_argument("--learning-rate",   type=float, default=None,
                        help="Learning rate override (default: preset value)")
    parser.add_argument("--encoder",         default=None,
                        help="Encoder checkpoint override for figurative training")
    parser.add_argument("--freeze-encoder",  action="store_true",
                        help="Freeze the encoder and only train the classification head (probing)")

    # figurative language detection
    parser.add_argument("--train-figurative",    action="store_true",
                        help="Train figurative classifier on VUA20 + MAGPIE + FLUTE")
    parser.add_argument("--figurative-experiment", default="tlm_base",
                        choices=list(FIGURATIVE_PRESETS),
                        help="Figurative experiment preset (default: tlm_base)")
    parser.add_argument("--predict-figurative",  action="store_true",
                        help="Run figurative detection on Bloomfield Cree sentences + idioms.txt")
    parser.add_argument("--figurative-checkpoint", default="KonradBRG/xlm-r-plains-cree-en-tlm-figurative",
                        help="Checkpoint for --predict-figurative / --eval-idioms")
    parser.add_argument("--figurative-output",   default="data/bloomfield_figurative.csv",
                        help="Output CSV for --predict-figurative")
    parser.add_argument("--input-file",          default="data/bloomfield_texts_sentences.csv",
                        help="Input CSV for --predict-figurative")

    # cross lingual knowledge distillation
    parser.add_argument("--distill-figurative",  action="store_true",
                        help="Cross-lingual knowledge distillation on the parallel corpus")
    parser.add_argument("--distill-mode",        default="clkd", choices=["align", "binary_kl", "clkd"],
                        help="Distillation mode (default: clkd)")
    parser.add_argument("--distill-checkpoint",  default="KonradBRG/xlm-mlm-100-1280-plains-cree-en-tlm",
                        help="Student model checkpoint")
    parser.add_argument("--distill-teacher",     default=None,
                        help="Frozen English teacher checkpoint for clkd mode")
    parser.add_argument("--distill-freeze-layers", type=int, default=0,
                        help="Freeze first N student layers during CLKD (0 = train all)")
    parser.add_argument("--distill-output",      default="data/figurative/distilled",
                        help="Output directory for distilled model")
    parser.add_argument("--distill-temperature", type=float, default=2.0,
                        help="Softening temperature for KL divergence (default: 2.0)")

    # evaluation
    parser.add_argument("--eval-idioms",     action="store_true",
                        help="Evaluate a figurative model on the Cree idiom golden test set")
    parser.add_argument("--idioms-file",     default="data/idioms.txt",
                        help="Path to the cree ||| english idioms file (default: data/idioms.txt)")

    args = parser.parse_args()

    if not any([args.scrape, args.scrape_edtekla, args.eda, args.annotate,
                args.split_sentences, args.fine_tune,
                args.train_figurative, args.predict_figurative,
                args.distill_figurative, args.eval_idioms]):
        parser.print_help()
        return

    if args.scrape:
        scrape()

    if args.scrape_edtekla:
        scrape_edtekla(args.edtekla_output, append=args.edtekla_append)

    if args.eda:
        eda()

    if args.annotate:
        annotate()

    if args.split_sentences:
        split_sentences(args.output, args.confidence)

    if args.fine_tune:
        ckpt = fine_tune(
            args.confidence, args.sentences_file, args.epochs, args.batch_size,
            args.hub_model_id, args.wandb_project, args.model_name,
            args.tlm_output_dir, args.grad_accum, args.max_length,
        )
        print(f"Checkpoint: {ckpt}")

    if args.train_figurative:
        ckpt = figurative_train(
            experiment     = args.figurative_experiment,
            epochs         = args.epochs,
            batch_size     = args.batch_size,
            learning_rate  = args.learning_rate,
            hub_model_id   = args.hub_model_id,
            wandb_project  = args.wandb_project,
            encoder        = args.encoder,
            freeze_encoder = args.freeze_encoder,
        )
        print(f"Checkpoint: {ckpt}")

    if args.predict_figurative:
        predict_figurative(
            checkpoint  = args.figurative_checkpoint,
            input_file  = args.input_file,
            output_file = args.figurative_output,
        )

    if args.distill_figurative:
        ckpt = figurative_distill(
            checkpoint         = args.distill_checkpoint,
            teacher_checkpoint = args.distill_teacher,
            freeze_n_layers    = args.distill_freeze_layers,
            mode               = args.distill_mode,
            corpus_file        = args.input_file,
            epochs             = args.epochs,
            batch_size         = args.batch_size,
            learning_rate      = args.learning_rate if args.learning_rate else 5e-6,
            temperature        = args.distill_temperature,
            hub_model_id       = args.hub_model_id,
            wandb_project      = args.wandb_project,
            output_dir         = args.distill_output,
        )
        print(f"Distilled model saved to: {ckpt}")

    if args.eval_idioms:
        figurative_eval_idioms(
            checkpoint  = args.figurative_checkpoint,
            idioms_file = args.idioms_file,
        )


if __name__ == "__main__":
    main()
