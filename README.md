Raspoznavanje uzoraka - Projekt
Akademska godina: 2013/2014
Tema 2. Raspoznavanje osoba na temelju "meke" biometrijske značajke - siluete osobe (i identifikacija)
======================================================
Autori:
Dorija Humski
Marija Mikulić
Dario Pavlović
Matija Šantl
Robert Vaser

kontakt: dario.pavlovic@fer.hr
link: https://github.com/mmikulic/ProjektRasUzo

Verzija: 1.O
Copyright: Nema.
======================================================
Program je testiran na Linux Mint distribuciji i Mac OS operativnim sustavima. Nema garancije da radi na bilo kojim drugima.
Također, program je isključivo podržan uz python verzije 2.7 јеr ОpenCV trenutno NE podražava Python verzije >= 3.
======================================================

Program implementira raspoznavanje osoba na temelju njihovih silueta. Program ima dva koraka, učenje klasifikatora
i testiranje, to jest isprobavanje.

Program pretpostavlja sljedeće:

Skup za učenje je direktorij u kojem se nalaze slike osoba. Primjer ovakvog direktorija su baza_2013 i baza_2014. 
Svaka osoba ima pohranjene fotografije u jedan poddirektorij. U tom poddirektoriju nalazi se direktorij 
pics/ i direktorij back/. U direktoriju pics/ nalaze se fotografije te osobe, a u direktoriju back/ pozadine. 
Svaka fotografija u direktoriju pics/ ima istoimenu odgovarajuću fotografiju pozadine u back/. 
Programu se zadaje put do početnog, korijenskog direktorija ovakve strukture (primjerice, <put_do_baza_2013>).
Također, program pretpostavlja da je ime osobe jednako imenu direktorija u kojem se nalaze pics/ i back/.
Na temelju ovoga se kasnije radi statistika točne klasifikacije. Primjerice, direktorij imena matija
bit će uzet kao ime osobe koja se nalazi na slikama u tom direktoriju.

Program se pokreće sa sljedećim obaveznim parametrima (za ostale, pokrenuti main.py s opcijom -h):
--path=<put_do_skupa_za_učenje>
--method=hu|granlund -- metoda izlučivanja značajki: Hu momenti ili Granlundovi koeficijenti
--threshold=<broj> -- prag koji prilikom izlučivanja siluete određuje koje točke postaju bijele (silueta) ili crne (pozadina)

Svi ostali parametri imaju pretpostavljene vrijednosti koje se mogu vidjeti pokretanjem main.py uz opciju -h.

Primjer osnovnog pokretanja programa:

python src/main.py --path=baza_2013/ --method=hu --threshold=50

Nakon što program nauči podatke koje dobije, moguće ga je testirati. Program će nakon faze učenja obavijestiti korisnika 
da je učenje gotovo i čekati da mu se na standardni ulaz preda putanja do fotografije, ime osobe te putanja do fotografije pozadine. 
Na temelju ovoga program će dati klasifikaciju i procijeniti je li ona točna uspoređujući ime osobe koje je producirao svaki
klasifikator s imenom koje mu se zadalo kao ispravno. Kad se želi završiti s programom, dovoljno je utipkati -1.

Na primjer:

python src/main.py --path=baza_2013/ --method=hu --threshold=50
Type in the path to a picture and its background. -1 to end.
Optionally, you can also put parameters if you selected the option when running.

test_baza_2014/dario/pics/dario1.JPG dario
test_baza_2014/dario/back/dario1.JPG
Loading test_baza/dario/pics/dario1.JPG
Loading test_baza/dario/back/dario1.JPG
Bayes result: robert
KNN result: robert
Tree result: mario

test_baza_2014/dorija/pics/dorija1.JPG dorija
test_baza_2014/dorija/back/dorija1.JPG
Loading test_baza/dorija/pics/dorija1.JPG
Loading test_baza/dorija/back/dorija1.JPG
Bayes result: robert
KNN result: robert
Tree result: dario

test_baza_2014/dorija/pics/dorija2.JPG dorija
test_baza_2014/dorija/back/dorija2.JPG
Loading test_baza/dorija/pics/dorija2.JPG
Loading test_baza/dorija/back/dorija2.JPG
Bayes result: robert
KNN result: marko
Tree result: robert
-1

Correctness:
Bayes: 0.09091
KNN: 0.18182
tree: 0.22727

Brojke i vrijednosti su samo ilustrativne u ovom slučaju. U direktoriju su pripremljene ulazne datoteke za testiranje
prilagođene skupovima koji se tu također nalaze tako da je moguće pokrenuti program i preusmjeriti neku od tih datoteka
na njegov standardni ulaz. On će ih pročitati i ispisati zbirne rezultate klasifikacije. paths.txt odgovaraju bazi iz 2013, 
a paths2.txt bazi iz 2014. Uloga paths3.txt datoteke detaljnije je opisana u dokumentaciji.

Na primjer:

python src/main.py --path=baza_2013/ --method=hu --threshold=50 < paths.txt

Postoje brojni drugi parametri vezani za klasifikatore i ostale mogućnosti programa, a svi se mogu vidjeti pokretanjem main.py
s opcijom -h ili u dokumentaciji.
