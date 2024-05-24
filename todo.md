- Recap Risultati
  - Bert (big boi)
  - DistilBert (pare sia peggio del naive bayes)
  - NaiveBayes... Not So Naive. (non è così naive)
  - RNNs (non esistono)
  - Zero Shot (non va)
- Documento Latex: struttura? (vedi dopo)
- Modelli da finire? (si bro le rnn)
- Stato del codice (tanti branch. davideb ...)
- Controllare 'get_data' (a quanto pare va bene così)
- Come misurare i risultati (f1 score)
- Valutare ipotesi della riduzione del Dataset:
  - Formulazione (vedi ipotesi)
  - Migliorare Analisi (84 è un numero a caso)
- Analizzare frasi classificate male. Perché 'gna 'a famo? (mettere le mani nella spazzatura)
- ...e il test set? (si ammettiamo che è un po' meme se i libri sono gli stessi del training set)
- Siamo stupidi: le frasi di un libro non possono sempre essere positivamente correlate alla corrente filosofica di riferimento. Per di più una dissertazione filosofica segue spesso i metodi retorici dell'induzione e della deduzione, incluso la negazione dell'assurdo. E noi trattiamo l'assurdo ... come positivo? cringe bro.
- Ma sto ensemble? (si fa? vediamo)
- Bert e DistilBert sono da valutare ... ma forse tutti so da valutare (evabbe)

---
# Documento Latex: struttura?
## Introduzione
si ecco qua il nostro lavoro vogliamo classificare quanto sei fascio
## Lavori precedenti (filosofo)
dunque c'è questo tizio vabbe ha dei risultati ridicolo
## Dataset Description
ecco qua il dataset che sto tizio ha raccolto veramente ridicolo sbilanciato insomma cambiamolo
### Contrastive Analisys
si perché allora gutenberg e tutto un po' ah guarda le nuvole di parole !
### Preprocessing
uaaa bello sto 30% del dataset zaccccc emmò non c'è più vabbe dai non piangere tanto non serviva 
## Metrica (F1-score MACRO AVG)
perché l'accuracy è #cringe. ma cos'è sta weighted macro average? forse è qualcosa che somiglia alla micro average. e allora tantovale fare la macro average. infatti facciamo quella. maronn
## La nostra ipotesi
Insomma ci aspettiamo che trainare e testare sul dataset ridotto sia il top del top, però aspettiamoci anche che magari avere più dati potrebbe essere meglio (però aumenta il rumore ... ma il rumore fa regolarizzazione? non è chiaro. dipende.), oppure che proprio siamo coglioni e le frasi piccole possano avere più significato del normale. In tal caso siamo fottuti e facciamo in tempo a ritirarci su un eremo lontano in emilia ma non tutti nello stesso (ce ne sono tanti).

alfa > beta,  
alfa > gamma,  
beta > delta,  
gamma > delta

## Modelli
### Naive Bayes
not so naive
### Bert
big boi
### DistilBert
smol boi
### Zero-Shot Learning
doesn't work
### RNNs
doesn't work (for now)
### Ensemble Learning
doesn't exist and may be long to train

## Risultati
qua ci stanno cinque tabelle, e per ogni modello mostriamo l'F1 score. E poi? sticazzi.
