"""The evaluated methods, their category, and the fixed score direction pi_m (Sec. III of the paper).
pi_m = +1: the native score increases with predicted utility. pi_m = -1: the native score measures a degradation
(embedding uncertainty, gradient magnitude, pruning drift) and is inverted. pi_m is never changed per attack."""

METHODS = [  # key, display name, category, pi_m
    ("faceqnet", "FaceQnet", "Supervised", +1),
    ("sdd_fiqa", "SDD-FIQA", "Supervised", +1),
    ("clib_fiqa", "CLIB-FIQA", "Supervised", +1),
    ("diffiqa_r", "DifFIQA(R)", "Unsupervised, non-FR-model", +1),
    ("ediffiqa_l", "eDifFIQA(L)", "Unsupervised, non-FR-model", +1),
    ("ser_fiq", "SER-FIQ", "Unsupervised, FR-probing", +1),
    ("grafiqs_l", "GraFIQs(L)", "Unsupervised, FR-probing", -1),
    ("froq", "FROQ", "Unsupervised, FR-probing", +1),
    ("vitnt_fiqa", "ViTNT-FIQA", "Unsupervised, FR-probing", +1),
    ("prefiqs", "PreFIQs", "Unsupervised, FR-probing", -1),
    ("attn_fiqa", "ATTN-FIQA", "Unsupervised, FR-probing", +1),
    ("magface", "MagFace", "Self-supervised (FR-integrated)", +1),
    ("pfe", "PFE", "Self-supervised (FR-integrated)", -1),
    ("cr_fiqa_l", "CR-FIQA(L)", "Self-supervised (FR-integrated)", +1),
    ("vit_fiqa_t", "ViT-FIQA(T)", "Self-supervised (FR-integrated)", +1),
    ("carpm_fiqa_l", "CARPM-FIQA(L)", "Self-supervised (FR-integrated)", +1),
    ("ex_fiqa_fw_t", "EX-FIQA-FW(T)", "Self-supervised (FR-integrated)", +1),
]
FIQA = [m[0] for m in METHODS]
REFERENCE = ("selfmad_pp", "SelfMAD++", "Trained detector, reference only", -1)   # native score = morph probability
NAME = {k: n for k, n, _, _ in METHODS + [REFERENCE]}
CATEGORY = {k: c for k, _, c, _ in METHODS + [REFERENCE]}
PI = {k: p for k, _, _, p in METHODS + [REFERENCE]}

ATTACKS = {
    "FRLL": ["FaceMorpher", "OpenCV", "Webmorph", "MIPGAN_I", "MIPGAN_II", "DCMorph", "MorDIFF", "Arc2Morph"],
    "FEI": ["C01_FaceMorpher", "C02_FaceFusion", "C03_FaceMorph_NTNU", "C05_UBOMorph", "C08_Twente_STASM", "C15_SURYS",
            "C16_Twente_Batskos", "C25_Improved_landmark_Twente", "C21_LADIMO_inspired", "C22_StyleDemorpher_inspired",
            "C24_Arc2Morph", "C26_StableMorph"],
}
