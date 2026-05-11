CLASS_NAMES = [
    "poison_ivy",
    "virginia_creeper",
    "red_maple",
    "white_oak",
    "tulip_poplar",
    "loblolly_pine",
    "eastern_red_cedar",
    "american_holly",
    "flowering_dogwood",
    "black_cherry",
]

CLASS_TO_IDX = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {index: class_name for class_name, index in CLASS_TO_IDX.items()}
CLASS_TO_SEARCH = [class_name.replace("_", " ") for class_name in CLASS_NAMES]
