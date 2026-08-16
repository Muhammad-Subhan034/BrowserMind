"""
Curated sentence corpus used for two purposes in the modelprep pipeline:

1. Validation set: each sentence gets a "ground truth" fp32 sentence embedding
   computed directly in Python/NumPy. The C++ tool re-derives the same
   embedding from the *quantized* (INT8) weights and reports the cosine
   similarity delta -- this is the accuracy number that goes in the README.

2. Demo corpus: the same sentences (as full text) ship inside the web app as
   the default document set, so the app has something meaningful to search
   and visualize even before a user uploads their own files.

Deliberately spans disjoint topics so that the embedding-space scatter plot
in the browser shows visible, explainable clusters rather than one blob.
"""

CORPUS: list[str] = [
    # -- astronomy / space --
    "The James Webb Space Telescope captures infrared light from the earliest galaxies.",
    "Saturn's rings are made mostly of ice particles with a smaller amount of rocky debris.",
    "A neutron star can spin hundreds of times per second without tearing itself apart.",
    "Voyager 1 is now the most distant human-made object, still transmitting from interstellar space.",
    "Black holes warp spacetime so strongly that not even light can escape past the event horizon.",
    "Mars has seasons because its rotational axis is tilted, much like Earth's.",
    "Astronomers use redshift to measure how quickly distant galaxies are moving away from us.",

    # -- cooking / food --
    "Searing meat at high heat creates a flavorful crust through the Maillard reaction.",
    "A good risotto needs constant stirring and warm stock added a ladle at a time.",
    "Sourdough bread rises using wild yeast and bacteria cultivated in a starter.",
    "Balancing acidity, sweetness, salt, and fat is the core skill behind most great sauces.",
    "Fermenting vegetables in brine preserves them while developing complex sour flavors.",
    "Resting a steak after cooking lets the juices redistribute evenly through the meat.",
    "Fresh basil loses much of its aroma when exposed to prolonged high heat.",

    # -- finance / economics --
    "Central banks raise interest rates to slow inflation by making borrowing more expensive.",
    "Diversifying a portfolio across asset classes reduces exposure to any single market shock.",
    "Compound interest means returns are earned on both the principal and prior gains.",
    "A yield curve inversion has historically preceded many economic recessions.",
    "Venture capital firms accept high failure rates in exchange for outsized returns on winners.",
    "Inflation erodes the purchasing power of cash held over long periods.",
    "Index funds track a market benchmark instead of relying on active stock picking.",

    # -- sports --
    "A marathon runner has to manage glycogen stores carefully to avoid hitting the wall.",
    "In chess, controlling the center squares early gives a lasting positional advantage.",
    "Sprinters explode out of the blocks using fast-twitch muscle fibers for rapid acceleration.",
    "A well-executed pick and roll creates a mismatch between a guard and a slower defender.",
    "Swimmers reduce drag by streamlining their body position off every turn.",
    "Home field advantage in soccer is partly explained by referee bias and crowd noise.",
    "Climbers use chalk to reduce sweat and improve grip friction on the rock.",

    # -- gardening / nature --
    "Companion planting pairs species like tomatoes and basil to deter pests naturally.",
    "Bees rely on flower color and ultraviolet nectar guides to find pollen efficiently.",
    "Composting kitchen scraps returns nitrogen and organic matter back into garden soil.",
    "Deciduous trees drop their leaves in autumn to conserve water through winter.",
    "Mycorrhizal fungi form symbiotic networks that help tree roots absorb nutrients.",
    "Pruning fruit trees in late winter encourages stronger growth in the spring.",
    "Migratory birds navigate using the Earth's magnetic field alongside visual landmarks.",

    # -- computing / software (also nicely self-referential for this project) --
    "A compute shader dispatches thousands of GPU threads in parallel workgroups.",
    "Quantization reduces model weights to lower precision to save memory and bandwidth.",
    "A hash map offers average constant time lookups by mapping keys to bucket indices.",
    "Garbage collection reclaims memory automatically once no references to an object remain.",
    "A binary search halves the remaining search space with every comparison.",
    "Caching avoids recomputation by storing the result of an expensive operation for reuse.",
    "A race condition occurs when two threads access shared memory without synchronization.",

    # -- history --
    "The printing press dramatically lowered the cost of reproducing written text.",
    "The Silk Road connected merchants across Asia, the Middle East, and Europe for centuries.",
    "The Industrial Revolution shifted economies from agrarian labor to factory production.",
    "Ancient Roman engineers built aqueducts using precise gravity-fed gradients over long distances.",
    "The invention of the steam engine transformed both manufacturing and transportation.",
    "Trade routes across the Mediterranean spread not just goods but also ideas and language.",

    # -- medicine / biology --
    "Antibiotics target bacterial processes and are ineffective against viral infections.",
    "Vaccines train the immune system to recognize a pathogen without causing the disease itself.",
    "DNA replication uses complementary base pairing to copy genetic information accurately.",
    "Sleep consolidates memories by strengthening connections formed during the day.",
    "The mitochondria generate cellular energy through a process called oxidative phosphorylation.",
    "Antibiotic resistance spreads faster when antibiotics are overprescribed or misused.",

    # -- music --
    "A minor key often evokes a more somber or melancholic feeling than a major key.",
    "Syncopation places rhythmic emphasis on beats that are normally unaccented.",
    "A capo lets a guitarist change key while keeping the same familiar chord shapes.",
    "Reverb simulates the natural reflections of sound within a physical space.",
    "Jazz improvisation relies on a shared harmonic framework that musicians interpret live.",
]
