from predict import multi_feature_predict

fasta_name = "output_1744891657.173759.fasta".split("fasta")[0]

multi_feature_predict(f"data/fasta/{fasta_name}fasta",
                      f"data/output/{fasta_name}csv", batch_size=2048)
