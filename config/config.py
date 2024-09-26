config = {
    'MODELS': {
        'ENTITY-TAGGING': r'./weights/entity_tagging'
    },

    'SERVICE-MANAGER': {
        'GLOBAL': '/application/kie_service/service_manager.ini',
        'LOCAL': './config/service_manager.ini',
    },

    'DEFAULTS': {
        'SAVE-DIR': './uploads',
        'ALLOWED-FORMATS': ('jpg', 'img'),
    },

    'OFF': {


        'REQUIRED-COLUMNS': [
            'code', 'product_name', 'image_nutrition_url', 'energy-kj_100g', 'energy-kcal_100g', 'energy_100g', 'energy-from-fat_100g', 'fat_100g',
            'saturated-fat_100g', 'butyric-acid_100g', 'caproic-acid_100g', 'caprylic-acid_100g', 'capric-acid_100g', 'lauric-acid_100g',
            'myristic-acid_100g', 'palmitic-acid_100g', 'stearic-acid_100g', 'arachidic-acid_100g', 'behenic-acid_100g', 'lignoceric-acid_100g',
            'cerotic-acid_100g', 'montanic-acid_100g', 'melissic-acid_100g', 'unsaturated-fat_100g', 'monounsaturated-fat_100g',
            'omega-9-fat_100g', 'polyunsaturated-fat_100g', 'omega-3-fat_100g', 'omega-6-fat_100g', 'alpha-linolenic-acid_100g',
            'eicosapentaenoic-acid_100g', 'docosahexaenoic-acid_100g', 'linoleic-acid_100g', 'arachidonic-acid_100g',
            'gamma-linolenic-acid_100g', 'dihomo-gamma-linolenic-acid_100g', 'oleic-acid_100g', 'elaidic-acid_100g', 'gondoic-acid_100g',
            'mead-acid_100g', 'erucic-acid_100g', 'nervonic-acid_100g', 'trans-fat_100g', 'cholesterol_100g', 'carbohydrates_100g',
            'sugars_100g', 'added-sugars_100g', 'sucrose_100g', 'glucose_100g', 'fructose_100g', 'lactose_100g', 'maltose_100g',
            'maltodextrins_100g', 'starch_100g', 'polyols_100g', 'erythritol_100g', 'fiber_100g', 'soluble-fiber_100g',
            'insoluble-fiber_100g', 'proteins_100g', 'casein_100g', 'serum-proteins_100g', 'nucleotides_100g', 'salt_100g',
            'added-salt_100g', 'sodium_100g', 'alcohol_100g', 'vitamin-a_100g', 'beta-carotene_100g', 'vitamin-d_100g', 'vitamin-e_100g',
            'vitamin-k_100g', 'vitamin-c_100g', 'vitamin-b1_100g', 'vitamin-b2_100g', 'vitamin-pp_100g', 'vitamin-b6_100g',
            'vitamin-b9_100g', 'folates_100g', 'vitamin-b12_100g', 'biotin_100g', 'pantothenic-acid_100g', 'silica_100g',
            'bicarbonate_100g', 'potassium_100g', 'chloride_100g', 'calcium_100g', 'phosphorus_100g', 'iron_100g', 'magnesium_100g',
            'zinc_100g', 'copper_100g', 'manganese_100g', 'fluoride_100g', 'selenium_100g', 'chromium_100g', 'molybdenum_100g', 'iodine_100g', 'caffeine_100g', 'taurine_100g', 'ph_100g',
            'fruits-vegetables-nuts_100g', 'fruits-vegetables-nuts-dried_100g', 'fruits-vegetables-nuts-estimate_100g',
            'fruits-vegetables-nuts-estimate-from-ingredients_100g',  'collagen-meat-protein-ratio_100g', 'cocoa_100g',
            'chlorophyl_100g', 'carbon-footprint_100g', 'carbon-footprint-from-meat-or-fish_100g',
            'nutrition-score-fr_100g', 'nutrition-score-uk_100g', 'glycemic-index_100g', 'water-hardness_100g', 'choline_100g',
            'phylloquinone_100g', 'beta-glucan_100g', 'inositol_100g', 'carnitine_100g', 'sulphate_100g', 'nitrate_100g', 'acidity_100g'],

        'IDENTIFIER-COLUMNS': ('product_name', 'code', 'image_nutrition_url'),
        'NUTRITION-COLUMNS': [
            'phosphorus', 'carbohydrates', 'vitamin-d', 'fruits-vegetables-nuts', 'fruits-vegetables-nuts-estimate-from-ingredients',
            'proteins', 'magnesium', 'chlorophyl', 'vitamin-c', 'cocoa', 'nutrition-score-fr', 'calcium', 'energy-kj',
            'trans-fat', 'fiber', 'vitamin-a', 'salt', 'starch', 'alcohol', 'potassium', 'monounsaturated-fat', 'sodium',
            'energy-kcal', 'copper', 'energy', 'sugars', 'added-sugars', 'polyunsaturated-fat',
            'carbon-footprint-from-meat-or-fish', 'fruits-vegetables-nuts-estimate', 'fat', 'caffeine', 'saturated-fat',
            'vitamin-b9', 'cholesterol', 'iron', 'zinc']

    }
}