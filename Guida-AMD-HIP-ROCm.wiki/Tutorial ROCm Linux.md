# Tutorial su come configurare un ambiente di sviluppo per eseguire programmi IA utilizzando ROCm

## Capitoli:

* ### [Premessa](#my-premessa)

* ### [A cosa serve questa guida](#my-a-cosa-serve-questa-guida)

* ### [A cosa non serve questa guida](#my-a-cosa-non-serve-questa-guida)

* ### [Prerequisiti](#my-prerequisiti)

* ### [Glossario e comandi di base](#my-glossario-e-comandi-di-base)

* ### [Creazione dell'ambiente tramite distrobox](#my-creazione-dell'ambiente-tramite-distrobox)

* ### [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

* ### [Utilizzo di un generico programma](#my-utilizzo-di-un-generico-programma)

* ### LLM

	* #### [Llamacpp](#my-llamacpp)
	
 	* #### [Koboldcpp](#my-koboldcpp)
 	
 	* #### [vllm](#my-vllm)
 	
* ### Immagini

	* #### [Stable Diffusion web UI](#my-stable-diffusion-web-ui)
	
	* #### [ComfyUI](#my-comfyui)

* ### Quantizzazione

	* #### [Bitsandbytes](#my-bitsandbytes)

	* #### [AutoAWQ](#my-autoawq)
	
	* #### [GPTQModel](#my-gptqmodel)

  	* #### [HQQ](#my-hqq)
	
	* #### [compressed-tensors](#my-compressed-tensors)

	* #### [FBGEMM](#my-fbgemm)

	* #### [torchao](#my-torchao)

	* #### [ExLlamaV2](#my-exllamav2)

	* #### [MLC](#my-mlc)

	* #### [llm-compressor](#my-llm-compressor)
	
* ### Librerie

	* #### [ONNX Runtime](#my-onnxruntime)
	
	* #### [Tensorflow](#my-tensorflow)
	
	* #### [JAX](#my-jax)
	
	* #### [CTranslate2](#my-ctranslate2) **FEEDBACK**
	
	* #### [Flash Attention](#my-flashattention) **NON PROPRIO FUNZIONANTE**
	
	* #### [xformers](#my-xformers) **NON FUNZIONANTE**
	
* ### Fine-tuning

	* #### [LLama-factory](#my-llama-factory)
	
	* #### [axolotl](#my-axolotl)
	
	* #### [lightning](#my-lightning)
	
	* #### [torchtune](#my-torchtune)

	* #### [fluxgym](#my-fluxgym)
	
	* #### [Unsloth](#my-unsloth) **NON FUNZIONANTE**
	
* ### [Link utili](#my-link-utili)

***

### <a id="my-premessa">Premessa:</a>
Questa guida verrà sicuramente ampliata e corretta nel tempo. In caso di problemi o se si vogliono suggerire modifiche siete fortemente invitati a farlo. Questa guida è fatta da persone per le persone.

### <a id="my-a-cosa-serve-questa-guida">A cosa serve questa guida:</a>

Alla fine di questo tutorial avrai creato un ambiente in cui potrai eseguire
programmi AI accelerandoli con la tua scheda AMD (per esempio utilizzando
stable diffusion per creare immagini). Questa guida è espressamente rivolta
a linux perchè per il momento è il luogo migliore in cui utilizzare l'AI
con gpu AMD.

### <a id="my-a-cosa-non-serve-questa-guida">A cosa non serve questa guida:</a>

Qui non imparerai ad utilizzare i suddetti programmi e senza una conoscenza più
o meno avanzata di informatica e di linux potresti non capire alcuni passaggi.
Spiegherò solo quello che ritengo fondamentale e che sia estremamente facile da
capire, per maggiori informazioni, per esempio sul funzionamento dei programmi,
linkerò in apposite sezioni siti e altre guide che reputo utili. Un altra cosa
da sottolineare è che questa guida riguarda esclusivamente il funzionamento di
ROCm su linux e non su Windows, i passaggi qui utilizzati non funzioneranno su
Windows.

### <a id="my-prerequisiti">Prerequisiti:</a>
* Una scheda AMD (possibilmente una scheda appartenente alle seguenti generazioni: 6000(rdna2)/7000(rdna3)/9000(rdna4). Il supporto per le APU è in forte sviluppo in questo momento (ROCm6.4.1) e in teoria sono supportare e ci sono anche risultati per quanto riguarda le performance. Per maggiori informazioni vedere la sezione link utili. **Questa guida presuppone l'utilizzo di sistemi con una singola scheda video discreta, sistemi con più gpu discrete non sono mai stati testati da me ma con accorgimenti dovrebbero funzionare. Nel caso basta aprire un Issue e discuterne.**
* Un ssd per installare linux. Si può procedere in 2 modi:
	* Ridimensionando il disco dove già è installato Windows.
	* Comprando un altro ssd (consiglio un tera, minimo 500 gb)
* Teoricamente è possibile utilizzare qualsiasi distribuzione linux ma i miei test sono stati fatti su [Fedora Linux](https://fedoraproject.org/), quindi se partite da 0 installate Fedora, nello specifico la versione KDE, altrimenti provate con la vostra distribuzione che state già utilizzando (non preoccupatevi, tutti i passaggi in questa guida non sono assolutamente pericolosi per il sistema operativo quindi non romperete la vostra installazione se utilizzate già linux). **In questa guida non si spiegherà come installare linux ma ci sono numerosissime guide anche in italiano su come farlo.**
* Conoscenza di Linux? In teoria non è necessaria perchè utilizzeremo pochi comandi molto semplici e autoesplicativi però ovviamente una conoscenza anche di base vi potrebbe aiutare
* Un po' di pazienza: purtroppo molti passaggi richiedono il download di molti file o altri tempi tecnici per installare software quindi preparatevi qualcosa da bere e da sgranocchiare!

### <a id="my-glossario">Glossario e comandi di base:</a>

Quanto segue è una sorta di spiegazione più o meno dettagliata di alcuni comandi e concetti presenti nella guida.

**Tutto quello che verrà detto in seguito non è preciso dal punto di vista tecnico perché semplificherò fino all'osso tutti i concetti!!!**

Il comando `cd` serve per navigare tra le cartelle: `cd [nome cartella]`, es: `cd llama.cpp` per entrare nella cartella mentre `cd ..` per andare nella cartella che contiene quella in cui ti trovi. Si può usare anche con il percorso completo della cartella `cd [percorso cartella]`, es: `cd /home/mario/llama.cpp`. Per avere il percorso completo di una cartella: tasto destro sulla cartella in questione e cliccare "copia posizione".

il comando `git clone` clona la così detta repository da internet per poterla utilizzare sul proprio pc mentre il comando 'git pull' aggiorna la repository locale alla versione madre che utilizza lo sviluppatore. Le repositories non sono altro che un insieme di file in cui si trovano programmi, librerie, etc.

I comandi `cmake` e `make` sono fondamentali perchè compilano il programma per farlo funzionare sul tuo computer. In parole povere se questi comandi restituiscono errore durante la compilazione non potrai utilizzare il programma.

Il comando `dnf` serve ad installare pacchetti in Fedora.

Il comando [`distrobox`](https://distrobox.it/) serve a gestire i container in maniera rapida, automatizzando un sacco di passaggi lunghi e noiosi. Nello specifico i comandi più importanti sono: `distrobox enter [nome container]` per entrare nel container, `distrobox rm -f [nome container]` per rimuovere il container e `distrobox create -i [immagine del container] -n [nome container]` per creare il container.

Il comando [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) serve a gestire gli ambienti virtuali di python. I comandi principali sono: `conda create -n [nome ambiente] python=[versione]` per creare un ambiente, `conda activate [nome ambiente]` per attivate l'ambiente e `conda remove -n [nome ambiente] --all` per rimuovere un ambiente. In generale anche se non è consigliato i pacchetti si installano con `pip install` e non con `conda install`. Per maggiori informazioni ci sono decine di guide su internet con tutti i comandi più avanzati.

Saranno presenti spesso le sigle gfx con numeri a seguire: queste sigle rappresentano le famiglie di schede video. Nello specifico gfx1030=rx6000/rdna2, gfx1100=rx7000/rdna3, gfx1201=rx9070, gfx1200=rx9060.

### <a id="my-creazione-dell'ambiente-tramite-distrobox">Creazione dell'ambiente tramite distrobox:</a>

Prima di procedere con qualsiasi cosa apri il terminale per eseguire il
seguente comando che aggiornerà il sistema operativo e tutti i suoi programmi:

```
sudo dnf upgrade -y --refresh
```

Successivamente installiamo distrobox, il programma che ci permetterà di
gestire tutto quello che riguarda ROCm tramite i container ufficiali di AMD:

```
sudo dnf install -y distrobox
```

Creiamo un container. Per chi non sappia cosa sia potete immaginarlo come un
"mini sistema operativo" che utilizzeremo perchè conterrà tutte le librerie e
il software di ROCm. Questo approccio ha più vantaggi ma i principali sono: nel
caso di problemi o di un aggiornamento di ROCm basterà eliminare il container e
crearlo da capo per ritornare ad un ambiente identico a quello consigliato da
AMD; è possibile avere più versioni di ROCm differenti in sistemi differenti da
testare in caso di problemi con versioni specifiche; è possibile installare ROCm su qualsiasi distribuzione Linux senza dover per forza installare Ubuntu o quelle supportate; aumenta la riproducibilità, cioè seguendo questa guida alla lettera la possibilità che qualcosa non funzioni solo a te e non ad altri è zero, quindi al contrario se funziona per una persona funziona per tutti. Il comando per creare il
container è il seguente (per installare una differente versione di ROCm basta
cambiare il numero della versione, in questo caso 6.4.1, con quello che volete,
per esempio 6.3.3,):

```
distrobox create -Y -i docker.io/rocm/dev-almalinux-8:6.4.1-complete -n almalinux-rocm
```

ed entriamo nel container appena creato chiamato almalinux-rocm:

```
distrobox enter almalinux-rocm
```

Entriamo nel vivo del tutorial facendo un rapido setup del container appena
creato. Non spiegherò i singoli passaggi ma farò un breve riassunto:

* Aggiungiamo i repo rpm fusion
* Installiamo git, miniconda, cmake, ccache, ffmpeg, make, bash-completition, gcc-toolset-9

```
# I comandi nelle sezioni vanno inseriti tutti insieme, tranne se specificato diversamente
sudo dnf install -y --nogpgcheck https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(rpm -E %rhel).noarch.rpm \
&& sudo dnf install -y --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-$(rpm -E %rhel).noarch.rpm \
&& sudo dnf install -y git git-lfs ffmpeg conda bash-completion cmake ccache make gcc-toolset-9*
```

Fatto! Non ci resta altro che entrare nel dettaglio per configurazioni specifiche per ogni famiglia di schede video. Il passaggio successivo è quello di inserire alcuni parametri in un file chiamato .bashrc nella cartella con il vostro nome utente, detta cartella home. Per fare questo chiudiamo il terminale
usato fin'ora e apriamone uno nuovo ed eseguiamo il comando nano che
viene utilizzato per modificare file di testo:

```
nano .bashrc
```

Nel file appena aperto scorri fino in fondo e incolla quanto segue, questi valori indicano che vogliamo utilizzare esclusivamente la prima gpu del computer così da evitare di utilizzare quella integrata, se anch'essa è AMD:

```
export HIP_VISIBLE_DEVICES="0"
export ROCR_VISIBLE_DEVICES="0"
export OMP_DEFAULT_DEVICE="0"
export GPU_DEVICE_ORDINAL="0"
```

Se avete anche schede Nvidia in teoria bisognerebbe "mascherarla" rendendola invisibile ai programmi. La procedura è documentata ma non è mai stata testata da me. Se vi trovate in una situazione in cui i programmi tentano di usare la scheda Nvidia anzichè quella AMD aprite un Issue.

Bisogna inserire un ulteriore parametro che però cambia in base alla
scheda video che si vuole utilizzare:

* Se si utilizza una 6000/rdna2:

```
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

* Se si utilizza una 7000/rdna3:

```
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Una volta inseriti per chiudere e salvare il file basta cliccare ctrl+c e successivamente premere s ed infine invio, come suggerito in basso dal programma stesso.

Se si utilizza una 9000/rdna4 c'è da fare un altro passaggio perchè purtroppo, almeno nella versione 6.4.1 di ROCm, la libreria ROCWMMA non funziona quindi è necessario installarla a mano.

Entriamo nel container almalinux-rocm:

```
distrobox enter almalinux-rocm
```

E installiamo la libreria

* Se si utilizza una 9070:

```
git clone https://github.com/ROCm/rocWMMA.git \
&& cd rocWMMA \
&& CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B build . -DROCWMMA_BUILD_TESTS=OFF -DROCWMMA_BUILD_SAMPLES=OFF -DGPU_TARGETS=gfx1201 \
&& cd build \
&& sudo make install
```

* Se si utilizza una 9060:

```
git clone https://github.com/ROCm/rocWMMA.git \
&& cd rocWMMA \
&& CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B build . -DROCWMMA_BUILD_TESTS=OFF -DROCWMMA_BUILD_SAMPLES=OFF -DGPU_TARGETS=gfx1200 \
&& cd build \
&& sudo make install
```

Et voilà! Abbiamo finito, la "difficilissa" installazione di ROCm.

### <a id="my-setup-pytorch-nel-container">Setup di pytorch nel container:</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

Pytorch è necessario in alcuni ambiti, principalmente per quanto riguarda la creazione di immagini e quando si lavora con l'audio, ed è effettivamente il vero "punto debole" di AMD, perché c'è sempre la possibilità che qualcosa non funzioni per bene e che si è costretti a provare svariate versioni fino a quando non si trova quella giusta con la possibilità che un aggiornamento al programma in uso ti rimetta di nuovo alla ricerca di versioni funzionanti. Ultimamente la situazione è migliorata tantissimo quindi sono fiducioso che se seguirai questi passi per installare pytorch non avrai problemi e tutto funzionerà al primo colpo.

Apriamo un terminale e iniziamo entrando nel container:

```
distrobox enter almalinux-rocm
```

In seguito creiamo quello che viene definito un ambiente virtuale con miniconda. Funziona un po' come un container ma molto più specifico perchè riguarda i pacchetti di python. L'ambiente virtuale ci permette, in caso di problemi, di eliminare l'ambiente e di ricrearne un altro in pochissimo tempo e di utilizzare più versioni di python diverse e con diverse versioni dei pacchetti. Così facendo potremo avere un ambiente specifico per ogni applicazione che funziona sempre e magari crearne di nuovi mano a mano più aggiornati per provare nuove funzioni senza modificare quello funzionante. Ma basta chiacchiere e creamo un ambiente virtuale chiamato py312 con python 3.12 installato:

```
conda create -y -n py312 python=3.12
```

Ed attiviamolo:

```
conda activate py312
```

Fatto ciò installiamo pytorch. Potete visitare il [sito ufficiale](https://pytorch.org/get-started/locally) di pytorch e seguendo la tabella instrallare i pacchetti con il comando consigliato. Nello specifico consiglio sempre di prenderli dalla versione nightly, a meno che non stiate utilizzando una versione particolamente vecchia di ROCm, perchè anche se potenzialmente meno stabili secondo il sito sono anche quelli più aggiornati e per esperienza funzionano sempre meglio. Ad oggi (ROCm6.4.1) il comando da eseguire è il seguente (Ovviamente cerca di abbinare la versione di ROCm nel comando seguente con la versione installata nel container, se non presente scaricare l'ultima versione nightly disponibile. Controllare il sito ufficiale per le versioni disponibili):

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4 \
&& pip install triton
```

Finito! Ora non resta che installare i vari programmi oppure, se si è capaci o si ha voglia di sperimentare, è possibile utilizzare direttamente la libreria transformers di Hugging Face in un programma python scritto da voi e accelerare i modelli con la gpu.

**Il resto della guida presuppone l'utilizzo di un generico ambiente virtuale denominato py312, nel caso di nomi diversi, per esempio se si desidera creare ambienti diversi per diverse applicazioni, bisogna sostituire py312 con il nome del vostro ambiente virtuale!!!**

### <a id="my-utilizzo-di-un-generico-programma">Utilizzo di un generico programma:</a>

Molti programmi che consiglio e che spiegherò come installare seguiranno questo workflow:

* `git clone` la repository su github corrispondente per scaricare il programma in una nuova cartella.
* `distrobox enter almalinux-rocm` per entrare nel container.
* Eseguire le varie procedure per installare tutte le librerie e pacchetti necessarie al funzionamento del programma o compilarlo.
* Eseguire il comando corrispondente per avviare il programma.

Una volta fatta l'installazione ed averlo testato per vedere se tutto funziona avviarlo le successive volte presuppone questi comandi:

* navigare con il gestore file fino a raggiungere la cartella contenente il programma.
* Aprire il terminale in quella cartella: tasto destro e clicca "apri terminale qui"
* `distrobox enter almalinux-rocm` per entrare nel container ed eventualmente attivare l'ambiente python.
* Eseguire il comando corrispondente per avviare il programma.

Per quanto riguarda l'ambiente virtuale di python il consiglio è di creare sempre ambienti virtuali separati per ogni programma da utilizzare (Llama-factory, axolotl, ecc.).

Questa è tutto. E no, non sto scherzando. Questa è la "gigantesca difficoltà nell'utilizzare gpu AMD" che si sente molto parlare su internet. Nella pratica è vero che AMD sbaglia su diverse cose (come verrà aggiunto in seguito, negli elenchi delle librerie non funzionanti) ma il grosso punto debole che rende difficile l'utilizzo di gpu AMD è la mancanza di guide online, contro la valanga di guide per schede Nvidia.

### <a id="my-llamacpp">Llamacpp</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

Aprire il terminale ed eseguire i seguenti comandi, verrà creata la cartella llama.cpp:

```
git clone https://github.com/ggml-org/llama.cpp.git \
&& cd llama.cpp
```

Entriamo nel container almalinux-rocm:

```
distrobox enter almalinux-rocm
```

Adesso bisogna compilarlo, nello specifico (potrebbero apparire scritte di avviso riguardo la compilazione ma si possono ignorare tutti. L'unico avviso importante è quello di errore in caso di problemi che verrà riportato al completamento della compilazione):

* Se si utilizza una 6000/rdna2:

```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DGGML_HIP=ON -DLLAMA_CURL=OFF -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -- -j$(nproc)
```

* Se si utilizza una 7000/rdna3

```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DGGML_HIP=ON -DLLAMA_CURL=OFF -DGGML_HIP_ROCWMMA_FATTN=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -- -j$(nproc)
```

* Se si utilizza una 9070

```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DGGML_HIP=ON -DLLAMA_CURL=OFF -DGGML_HIP_ROCWMMA_FATTN=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -- -j$(nproc)
```

* Se si utilizza una 9060

```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DGGML_HIP=ON -DLLAMA_CURL=OFF -DGGML_HIP_ROCWMMA_FATTN=ON -DAMDGPU_TARGETS=gfx1200 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -- -j$(nproc)
```

Per aggiornare llama.cpp eseguire i seguenti comandi con un terminale aperto nella cartella llama.cpp:

```
git pull \
&& rm -r build
```

E ripetere le istruzioni per compilare il codice.

### <a id="my-koboldcpp">Koboldcpp</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

Aprire il terminale ed eseguire i seguenti comandi, verrà creata la cartella koboldcpp:

```
git clone https://github.com/LostRuins/koboldcpp.git \
&& cd koboldcpp \
&& sudo dnf install -y python3.12-tkinter \
&& pip3.12 install --no-input customtkinter
```

Entriamo nel container almalinux-rocm:

```
distrobox enter almalinux-rocm
```

Adesso bisogna compilarlo, nello specifico (potrebbero apparire scritte di avviso riguardo la compilazione ma si possono ignorare tutti. L'unico avviso importante è quello di errore in caso di problemi che verrà riportato al completamento della compilazione):

* Se si utilizza una 6000/rdna2:

```
make LLAMA_HIPBLAS=1 GPU_TARGETS=gfx1030 -j$(nproc)
```

* Se si utilizza una 7000/rdna3:

```
make LLAMA_HIPBLAS=1 GPU_TARGETS=gfx1100 -j$(nproc)
```

* Se si utilizza una 9070:

```
make LLAMA_HIPBLAS=1 GPU_TARGETS=gfx1201 -j$(nproc)
```

* Se si utilizza una 9060:

```
make LLAMA_HIPBLAS=1 GPU_TARGETS=gfx1200 -j$(nproc)
```


Installiamo alcuni pacchetti (in caso di aggiornamento non è necessario ripetere questo passaggio ma solo quelli precedenti):

```
sudo dnf install -y python3.12-tkinter \
&& pip3.12 install --no-input customtkinter
```

Infine per lanciare kobolcpp:

```
python3.12 koboldcpp.py
```

Per aggiornare koboldcpp eseguire i seguenti comandi con un terminale aperto nella cartella koboldcpp:
```
git pull \
&& make clean
```

E ripetere le istruzioni per compilare il codice.

### <a id="my-vllm">vllm</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

Considero vllm uno strumento molto avanzato per il serving di llm quindi in realtà non rientra tra le librerie che un semplice utilizzatore di programmi AI ha bisogno ma essendo un pacchetto legato a Llama-Factory spiegherò come installarlo. Per un utilizzo più estensivo, cioè se si desidera utilizzare principalmente questa libreria, si consiglia di utilizzare i [docker ufficiali forniti da AMD](https://hub.docker.com/u/rocm?page=1&search=vllm). Dalla wiki ufficiale di vllm viene suggerito di installare triton per il supporto di flash attention, si consiglia di dare un occhiata alla parte di guida che riguarda [flash attention](#my-flashattention).

Iniziamo clonando la repository:

```
git clone https://github.com/ROCm/vllm.git \
&& cd vllm
```

Entriamo nel container almalinux-rocm e installiamo una versione differente di gcc perché almalinux ne utilizza una troppo vecchia(nel caso si stesse utilizzando un container Ubuntu la versione di gcc dovrebbe essere già sufficentemente alta). La guida mostra i comandi con il generico ambiente virtuale py312 (quello creato durante l'installazione di PyTorch) ma consiglio di creare specifici ambienti virtuali per ogni programma:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# Riavviamo la shell per usare momentaneamente gcc-9 come compilatore di default
scl enable gcc-toolset-9 bash

# Attiviamo l'ambiente virtuale py312
conda activate py312
```

Installiamo alcuni pacchetti obbligatori:

```
pip install -r requirements/rocm.txt
```


* Se si utilizza una 6000/rdna2:

```
PYTORCH_ROCM_ARCH="gfx1030" python setup.py develop
```

* Se si utilizza una 7000/rdna3:

```
PYTORCH_ROCM_ARCH="gfx1100" python setup.py develop
```

* Se si utilizza una 9070:

```
PYTORCH_ROCM_ARCH="gfx1201" python setup.py develop
```

* Se si utilizza una 9060:

```
PYTORCH_ROCM_ARCH="gfx1200" python setup.py develop
```

L'installazione di vllm per AMD non utilizzando il docker ufficiale è sconsigliata dagli sviluppatori quindi potrebbero esserci problemi.

### <a id="my-stable-diffusion-web-ui">Stable Diffusion web UI</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

Apriamo un terminale e lanciamo questi comandi con la conseguente creazione della cartella stable-diffusion-webui:

```
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
```

Navigiamo nella cartella:

```
cd stable-diffusion-webui
```

Entriamo nel container almalinux-rocm e attiviamo l'ambiente virtuale creato prima con pytorch installato. La guida mostra i comandi con il generico ambiente virtuale py312 (quello creato durante l'installazione di PyTorch) ma consiglio di creare specifici ambienti virtuali per ogni programma:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# Attiviamo l'ambiente virtuale py312
conda activate py312

# Installiamo le dipendenze
pip install -r requirements.txt
```

Per eseguire stable-diffusion-webui:

```
python launch.py
```

Se funzionava tutto con il comando precedente prova con questo che utilizza un'implementazione di flash attention tramite triton. Se non stai capendo quello che sto dicendo flash attention **in teoria** migliora le performance e riduce il carico sulla memoria della scheda video. Dico in teoria perché su AMD potrebbe: o non funzionare, funzionare in parte oppure funzionare appieno. Sulla mia RX 9070 XT funziona completamente ma essendo una funzione sperimentale potrebbe dare problemi.

```
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python launch.py
```

Per aggiornare Stable Diffusion web UI eseguire il seguente comando con un terminale aperto nella cartella stable-diffusion-webui:

```
git pull
```

E ripetere i comandi riguardo l'installazione.

**In generale in ogni programma che usa una versione abbastanza recente di pytorch (se non erro 2.6 in su) può essere abilitato flash attention tramite triton**

### <a id="my-comfyui">Comfyui</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell'ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

Cloniamo la repository da Github, verrà creata la cartella ComfyUI:

```
git clone https://github.com/comfyanonymous/ComfyUI.git
```

Navigiamo nella cartella:

```
cd ComfyUI
```

Entriamo nel container almalinux-rocm e attiviamo l'ambiente virtuale creato prima con pytorch installato. La guida mostra i comandi con il generico ambiente virtuale py312 (quello creato durante l'installazione di PyTorch) ma consiglio di creare specifici ambienti virtuali per ogni programma:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# e attiviamo l'ambiente virtuale py312
conda activate py312

# Installiamo le dipendenze
pip install -r requirements.txt
```

Per eseguire ComfyUI:

```
python main.py --use-pytorch-cross-attention
```

Come per Stable Diffusion web UI si può provare ad utilizzare flash attention con triton per migliori performance. Se ci sono problemi tornare al comando precedente:

```
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
```

Per aggiornare ComfyUI eseguire il seguente comando con un terminale aperto nella cartella ComfyUI:

```
git pull
```

E ripetere i comandi riguardo l'installazione.

### <a id="my-bitsandbytes">Bitsandbytes</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell'ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

**Prerequisiti:**
* La versione di pytorch installata deve coincidere con la versione di ROCm installata.

Procediamo con la clonazione della repository ma da un ramo specifico dove è abilitato il supporto alle gpu AMD, sulla carta solo alcune schede sono supportate (gfx90a e gfx1100) ma un tentativo anche con gfx1030 lo consiglio caldamente. Una guida ufficiale è presente a [questo link](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend) direttamente sul sito di Hugging Face ma la modificherò leggermente per essere più chiara. I comandi sono i seguenti:

```
git clone --recurse https://github.com/ROCm/bitsandbytes.git \
&& cd bitsandbytes \
&& git checkout rocm_enabled_multi_backend
```

Entriamo nel container almalinux-rocm e attiviamo l'ambiente virtuale creato prima con pytorch installato:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# Attiviamo l'ambiente virtuale py312
conda activate py312

# Installiamo le dipendenze
pip install -r requirements-dev.txt
```

Adesso bisogna compilarlo, nello specifico (potrebbero apparire scritte di avviso riguardo la compilazione ma si possono ignorare tutti. L'unico avviso importante è quello di errore in caso di problemi che verrà riportato al completamento della compilazione):

* Se si utilizza una 6000/rdna2:

```
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1030" -S .
```

* Se si utilizza una 7000/rdna3:

```
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1100" -S .
```

* Se si utilizza una 9070:

```
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1201" -S .
```

* Se si utilizza una 9060:

```
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1200" -S .
```


Successivamente eseguire:

```
make -j$(nproc) \
&& pip install .
```

Per aggiornare la libreria purtroppo è necessario eliminare la cartella e ripetere la procedura da capo.

### <a id="my-autoawq">AutoAWQ</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312

# Infine installiamo il pacchetto
pip install autoawq
```

### <a id="my-gptqmodel">GPTQModel</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312

# Infine installiamo il pacchetto
pip install gptqmodel --no-build-isolation
```


### <a id="my-hqq">HQQ</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312

# Infine installiamo il pacchetto
pip install hqq
```

### <a id="my-compressed-tensors">compressed-tensors</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312

# Infine installiamo il pacchetto
pip install compressed-tensors
```

### <a id="my-fbgemm">FBGEMM</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312

# Infine installiamo il pacchetto
pip install --pre fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```

### <a id="my-torchao">torchao</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312

# Infine installiamo il pacchetto
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```

### <a id="my-exllamav2">ExLlamaV2</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

Entriamo nel container almalinux-rocm e utilizziamo una versione differente di gcc perché almalinux di base ne utilizza una troppo vecchia(nel caso si stesse utilizzando un container Ubuntu la versione di gcc dovrebbe essere già sufficentemente alta). La guida mostra i comandi con il generico ambiente virtuale py312 (quello creato durante l'installazione di PyTorch) ma consiglio di creare specifici ambienti virtuali per ogni programma:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# Riavviamo la shell per usare momentaneamente gcc-9 come compilatore di default
scl enable gcc-toolset-9 bash

# Attiviamo l'ambiente virtuale py312
conda activate py312
```

Cloniamo la repository e installiamo il pacchetto:

```
git clone https://github.com/turboderp/exllamav2 \
&& cd exllamav2 \
&& pip install -r requirements.txt \
&& pip install .
```

### <a id="my-mlc">MLC</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312

# Infine installiamo il pacchetto
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-rocm62 mlc-ai-nightly-rocm62
```

**La compilazione non funziona per ora quindi quanto segue non è funzionante**

Entriamo nel container almalinux-rocm e attiviamo l'ambiente virtuale:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# Attiviamo l'ambiente virtuale py312
conda activate py312
```

Cloniamo la repository e installiamo il pacchetto. La configurazione è guidata e dovrete seguire i pochi e semplici passaggi rispondendo Y o N alle domande per selezionare il corretto backend. Nello specifico alla prima domanda riguardo TVM_SOURCE_DIR premere invio senza specificare niente, successivamente rispondere con `n` alle domande riguardanti CUDA, Vulkan, Metal e OpenCL mentre rispondere con `y` quando comparirà ROCm:

```
# Da eseguire in ordine uno alla volta

# Cloniamo la repository
git clone --recursive https://github.com/mlc-ai/mlc-llm.git && cd mlc-llm/

# Creiamo la cartella build
mkdir -p build && cd build

# Creiamo il file di configurazione con questo comando appariranno le domande
python ../cmake/gen_cmake_config.py

# Installiamo cargo
sudo dnf install -y cargo

# Compiliamo
cmake .. && cmake --build . --parallel $(nproc) && cd ..
```

### <a id="my-llm-compressor">llm-compressor</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312

# Infine installiamo il pacchetto
pip install llmcompressor
```

### <a id="my-onnxruntime">ONNX Runtime</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

Installiamo il pacchetto direttamente dalla repository ufficiale di AMD:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312
```

Si possono installare sia il pacchetto per l'inferenza che per il training.

* Inferenza:

```
pip install onnxruntime_rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/
```

* Training:

```
pip install onnxruntime_training -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/
```

### <a id="my-tensorflow">Tensorflow</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

Installiamo il pacchetto direttamente dalla repository ufficiale di AMD:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312
```

Si possono installare sia la versione base che nightly.

* Base:

```
pip install tensorflow_rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/
```

* Nightly:

```
pip install tf_nightly_rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/
```

### <a id="my-jax">JAX</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

Installiamo il pacchetto direttamente dalla repository ufficiale di AMD:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312

# Infine installiamo i pacchetti
pip install \
https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/jax-0.4.35-py3-none-any.whl \
https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/jax_rocm60_pjrt-0.4.35-py3-none-manylinux_2_28_x86_64.whl \
https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/jax_rocm60_plugin-0.4.35-cp312-cp312-manylinux_2_28_x86_64.whl \
https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/jaxlib-0.4.35-cp312-cp312-manylinux_2_28_x86_64.whl
```

### <a id="my-ctranslate2">CTranslate2</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

**Purtroppo non sono riuscito a farlo funzionare ma c'è una guida che ha portato maggiore fortuna ad altri. Di seguito viene riportato quello che ho provato io fallendo miseramente. Se qualcuno riesce a far funzionare tutto, perfavore segnalatelo**

https://github.com/OpenNMT/CTranslate2/issues/1072

https://github.com/arlo-phoenix/CTranslate2-rocm/blob/rocm/README_ROCM.md

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312
```

Cloniamo la repository:

```
git clone https://github.com/arlo-phoenix/CTranslate2-rocm.git --recurse-submodules \
&& cd CTranslate2-rocm
```

Inseriamo tutti i comandi per compilare tutto.


* Se si utilizza una 6000/rdna2:

```
CLANG_CMAKE_CXX_COMPILER=amdclang++ CXX=amdclang++ HIPCXX="$(hipconfig -l)/amdclang++" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DBUILD_TESTS=ON -DWITH_CUDNN=ON
```

* Se si utilizza una 7000/rdna3:

```
CLANG_CMAKE_CXX_COMPILER=amdclang++ CXX=amdclang++ HIPCXX="$(hipconfig -l)/amdclang++" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DBUILD_TESTS=ON -DWITH_CUDNN=ON
```

* Se si utilizza una 9070:

```
CLANG_CMAKE_CXX_COMPILER=amdclang++ CXX=amdclang++ HIPCXX="$(hipconfig -l)/amdclang++" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBUILD_TESTS=ON -DWITH_CUDNN=ON
```

* Se si utilizza una 9060:

```
CLANG_CMAKE_CXX_COMPILER=amdclang++ CXX=amdclang++ HIPCXX="$(hipconfig -l)/amdclang++" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DBUILD_TESTS=ON -DWITH_CUDNN=ON
```

Successivamente:

```
cmake --build build -- -j16 \
&& cd build \
&& cmake --install . --prefix $CONDA_PREFIX
```

Finalmente installiamo:

```
&& cmake --install . --prefix $CONDA_PREFIX
```

I test vengono eseguiti correttamente come scritto nel file README ma non riesco ad usare faster-whisper.

### <a id="my-flashattention">Flash Attention NON PROPRIO FUNZIONANTE</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

Per il momento (ROCm6.4.1) ci sono due backend per flash attention per AMD: Triton e composable_kernel. Triton supporta le architetture RDNA e CDNA (le Instinct di AMD) mentre composable_kernel funziona solo con CDNA. Ovviamente in questa guida si fa riferimento esclusivamente a RDNA quindi procediamo con Triton:

```
git clone https://github.com/ROCm/flash-attention.git \
&& cd flash-attention \
&& git checkout main_perf
```

Entriamo nel container almalinux-rocm e attiviamo l'ambiente virtuale creato prima con pytorch installato:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# e attiviamo l'ambiente virtuale py312
conda activate py312
```

Installiamo flash attention

```
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install
```

Poi va aggiunto al comando da eseguire con flash attention abilitato (oppure al file .bashrc per averlo sempre abilitato).

```
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
```

Per il momento flash attention non è da considerarsi attendibile come libreria: potrebbe non funzionare, funzionare occasionalmente o magari anche ridurre le prestazioni anzichè aumentarle. I feedback in merito sono estremamente utili.

### <a id="my-xformers">xformers</a>

Questa libraria è semplicemente non funzionante su RDNA perchè utilizza composable_kernel. Per colpa della mancanza di questa libreria non funziona Unsloth e Nvidia ha un vantaggio nella generazione di immagini. Sarebbe accettabile se la libreria fosse vecchia di un paio di mesi o un anno ma sono 3 anni che questa barzelletta va avanti. Riporterò di seguito alcuni link significativi:

https://github.com/ROCm/xformers/issues/9

https://github.com/ROCm/composable_kernel/issues/1171

https://github.com/ROCm/composable_kernel/issues/1434

https://github.com/ROCm/xformers/issues/17

https://github.com/ROCm/composable_kernel/issues/1958

### <a id="my-llama-factory">Llama-factory</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

**È fortemente consigliato creare un ambiente virtuale appositamente per Llama-facotory**

Iniziamo entrando nell'ambiente virtuale:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# e attiviamo l'ambiente virtuale py312
conda activate py312
```

Cloniamo la repository e installiamo Llama-factory:

```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git \
&& cd LLaMA-Factory \
&& pip install -e ".[torch,metrics]" --no-build-isolation
```

### <a id="my-axolotl">axolotl</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

**È fortemente consigliato creare un ambiente virtuale appositamente per axolotl**

Iniziamo entrando nell'ambiente virtuale:

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# e attiviamo l'ambiente virtuale py312
conda activate py312
```

Teoricamente è possibile installare axolotl con l'argomento flash-attn dopo aver installato flash attention in [questo modo](#my-flashattention) (i feedback sono ben accetti). Per quanto riguarda l'argomento deepspeed nella guida ufficiale dello sviluppatore non viene inserito ma farei dei test abilitandolo visto che per deepspeed ci sono stati diversi contributi per farlo funzionare su AMD, tentar non nuoce. Nel peggiore dei casi eseguire l'ultimo comando senza i due argomenti:

```
git clone https://github.com/axolotl-ai-cloud/axolotl \
&& cd axolotl \
&& pip install packaging ninja \
&& pip install --no-build-isolation -e .[,deepspeed]
```

Fatto ciò installiamo Bitsandbytes in [questo modo](#my-bitsandbytes) e flash-attention in [questo modo](#my-flashattention) per farlo funzionare su schede AMD.

### <a id="my-lightning">lightning</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312

# Infine installiamo il pacchetto
python -m pip install lightning
```

### <a id="my-torchtune">torchtune</a>

**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

* [Setup di pytorch nel container](#my-setup-pytorch-nel-container)

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312

# Infine installiamo il pacchetto
pip install --pre torchao torchtune --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```

### <a id="my-fluxgym">fluxgym</a>


**Presuppone che tu abbia seguito**:

* [Creazione dell'ambiente tramite distrobox](#my-creazione-dell-ambiente-tramite-distrobox)

**Ho personalmente modificato il programma per farlo eseguire su schede AMD. Sono riuscito a creare un modello ma c'è bisogno di fare ulteriori test. Ogni feedback è prezioso!!!**

[Link alla repository](https://github.com/wasd-tech/fluxgym-rocm)

[Link a sd-scripts-rocm](https://github.com/wasd-tech/sd-scripts-rocm)

**È fortemente consigliato creare un ambiente virtuale appositamente per fluxgym. Di seguito i comandi daranno per scontato l'utilizzo di un generico ambiente virtuale chiamato py312.**

**Python==3.12 obbligatoriamente e l'ambiente virtuale NON deve contenere torch o altro.**

```
# Da eseguire in ordine uno alla volta

# Ovviamente entriamo nel container almalinux-rocm
distrobox enter almalinux-rocm

# E attiviamo l'ambiente virtuale py312
conda activate py312
```

Cloniamo la repository principale e quella secondaria:

```
git clone https://github.com/wasd-tech/fluxgym-rocm \
&& cd fluxgym \
&& git clone -b sd3 https://github.com/wasd-tech/sd-scripts-rocm.git sd-scripts
```

Navigiamo nella cartella sd-scripts e installiamo i requisiti di sd-scripts

```
cd sd-scripts \
&& pip install -r requirements_amd.txt
```

Torniamo indietro e installiamo i requisiti di fluxgym

```
cd .. \
&& pip install -r requirements_amd.txt
```

Infine installiamo PyTorch sovrascrivendo la versione troppo vecchia usata dal programma (lo consiglia l'installazione di fluxgym in generale, per l'hardware nuovo è sicuramente obbligatorio)

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```

### <a id="my-unsloth">Unsloth</a>

Purtroppo non funziona, vedi [xformers](#my-xformers)

### <a id="my-link-utili">Link utili</a>

https://llm-tracker.info/howto/AMD-GPUs: La guida che ha inspirato questa opera.

https://initialxy.com/lesson/2025/02/15/fine-tuning-mistral-small: Fine-tuning di un modello con una scheda AMD.

https://github.com/ROCm/TheRock: Il progetto di AMD per creare un sistema automatico per la compilazione di tutte le librerie. Da tenere d'occhio perchè fornisce molte informazioni utili soprattutto per tutto quello che non è ufficialmente supportato.

https://rocm.docs.amd.com/en/latest/index.html: Documentazione ufficiale di ROCm. Molto utile in alcuni casi. Utili anche i vari blog che spiegano come utilizzare le schede AMD in alcuni contesti.

https://hub.docker.com/u/rocm: I docker ROCm ufficiali di AMD. Credo che sia il modo migliore e più affidabile di usare software ROCm ma sono molto specifici quindi non sempre si trova quello che si vuole.

https://repo.radeon.com/: La repository ufficiale di AMD

https://repo.radeon.com/rocm/manylinux/: La parte di repository di AMD interessante. Selezionando la versione è possibile trovare i pacchetti python ufficiali.
