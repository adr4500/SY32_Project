### Projet SY32 Apprentissage - Adrien Hervé, Constant Roussel

Pour utiliser ce détécteur d'écocup, veuillez placer les fichiers comme suit :

```bash
├── main.py
├── ...
└── dataset-original
    ├── test
    │   ├── 000.jpg
    │   ├── 001.jpg
    │   ├── ...
    │   └── 149.jpg
    └── train
        ├── images
        │   ├── neg
        │   │   ├── abigotte_neg_001.jpg
        │   │   ├── abigotte_neg_002.jpg
        │   │   ├── ...
        │   │   └── yboucher_neg_005.jpg
        │   └── pos
        │       ├── abigotte_pos_001.jpg
        │       ├── abigotte_pos_002.jpg
        │       ├── ...
        │       └── yboucher_pos_010.jpg
        └── labels_csv
            ├── abigotte_pos_001.csv
            ├── abigotte_pos_002.csv
            ├── ...
            └── yboucher_pos_010.csv
```

Ensuite, lancez le fichier main.py (l'execution de ce fichier est très longue , en particulier le second entrainement (plusieurs heures !), mais vous pouvez observer son évolution sur le terminal)