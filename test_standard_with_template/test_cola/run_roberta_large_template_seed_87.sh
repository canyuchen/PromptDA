




python ../../fine_tune_mlm_few_shot_roberta_standard_with_template.py  --student_model "../../data/model/roberta-large/" \
                                                            --data_dir "../../data/k-shot/CoLA/8-87/" \
                                                            --output_dir "../../output/dataset_num_8_fine_tune_mlm_few_shot_3_label_word_batch_size_4_bert-large-uncased/" \
                                                            --do_lower_case \
                                                            --learning_rate 3e-6  \
                                                            --num_train_epochs  10  \
                                                            --eval_step 5 \
                                                            --max_seq_length 128 \
                                                            --train_batch_size 4 \
                                                            --dataset_num 8 \
                                                            --data_seed 87 \
                                                            --task_name CoLA



