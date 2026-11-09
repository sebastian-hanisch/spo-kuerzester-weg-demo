# Der kürzeste Weg und der SPO-Verlust: der Löser in der Schleife – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-spo-kuerzester-weg-demo.streamlit.app/)** -->

Stück der Reihe **Entscheidungsorientiertes Lernen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vehikel ist ein abstraktes **5 × 5 Gitter** (Wege von links oben nach rechts unten, 40 Bögen, 70 Wege): Aus Merkmalen sollen Bogenkosten vorhergesagt werden, und ein exakter Löser (eine Rückwärtsrechnung, das Gitter ist azyklisch) wählt daraus den kürzesten Weg. Die Demo trainiert dasselbe lineare Modell auf zwei Arten, mit **Kleinsten Quadraten (KQ)** auf dem Fehler der Kosten und mit **SPO+** auf dem Regret des gewählten Weges, bei dem der **Löser in jeder Trainingsrunde läuft**.

## Warum dieses Problem

Wer Kosten vorhersagt und dann optimiert, bewertet das Modell meist nach dem Fehler der Kosten. Für die Entscheidung zählt aber nur, ob der gewählte Weg gut ist: Ein Modell kann die Kosten schlecht treffen und trotzdem den richtigen Weg wählen, und umgekehrt. Die Kernfrage der Demo: **Rettet SPO+ ein falsches lineares Modell, und wann nicht?**

- **Bestes Regret, schlechteste Prognose:** In der Welt mit Grad 6 ohne Rauschen und n = 5000 hat SPO+ ein Viertel des Regrets von KQ (Median 0,5 gegen 2,0 %), aber einen 1,8-fachen Quadratfehler je Bogen (1,39 gegen 0,76). Der Zufallswald der Vormessung (andere Seeds und Zufallsquelle) hat dort mit 0,29 den kleinsten Quadratfehler, sein Regret liegt bei 0,8 %.
- **Gewinn nur bei nichtlinearer Wahrheit:** Bei Grad 1 hat KQ ohne Rauschen einen Regret von 0,0 % und SPO+ ist in keinem der 20 Seeds besser. Ab Grad 4 senkt SPO+ ohne Rauschen bei n = 1000 den Regret in allen 20 Seeds.
- **Es gilt nicht überall:** Mit Rauschen schließt SPO+ nur einen Teil der Lücke zwischen KQ und Orakel, und bei wenig Daten (n = 100) liegt SPO+ mit Rauschen in den Welten mit Grad 1, 2 und 4 hinter KQ. Gegen eine flexible Modellklasse (Vormessung mit Gauß-Kern-Ridge, Grad 6) liegt SPO+ ohne Rauschen klar zurück; mit Rauschen ist der Abstand klein und nicht in jeder Zelle zugunsten der flexiblen Klasse. Der Satz der Demo ist deshalb eng: SPO+ rettet das falsche **lineare** Modell, unter diesen Bedingungen; er sagt nicht „SPO+ ist besser“.

## Modell

**Welt** (aus dem Gedächtnis nach dem Schema der SPO-Literatur nachgebaut, nicht gegen sie geprüft): Merkmale $x \sim \mathcal N(0, I_5)$; eine Gewichtsmatrix $B \in \{0,1\}^{40 \times 5}$ mit fairen Münzwürfen; mittlere Kosten $m_j(x) = ((Bx)_j/\sqrt{5} + 3)^{\deg} / 3{,}5^{\deg} + 1$; beobachtete Kosten $c_j = m_j(x)\,\varepsilon_j$ mit $\varepsilon_j \sim U[1 - e, 1 + e]$. Grad 1 ist linear in $x$ (ein lineares Modell kann die Wahrheit exakt abbilden), Grad 2, 4 und 6 weichen zunehmend davon ab. $e$ ist die Rauschstufe (0 oder 0,5 in der App). Das **Orakel** kennt $m(x)$; bei $e = 0$ hat es Regret 0.

**Löser.** Rückwärtsrechnung über die 25 Knoten, vektorisiert über alle Fälle. Gleichstand: am Knoten „rechts“ vor „unten“ (das ist der lexikographisch kleinste optimale Weg bei der Aufzählung mit R vor D).

**Verluste.** Mit wahren Kosten $c$, vorhergesagten $\hat c$, dem unter $\hat c$ gewählten Weg $w(\hat c)$ und dem besten Weg $w^*(c)$ mit Wert $z^*(c)$:

- SPO-Verlust (Regret): $c^\top w(\hat c) - z^*(c)$, stückweise konstant in $\hat c$.
- SPO+: $\max_w (c - 2\hat c)^\top w + 2\hat c^\top w^*(c) - z^*(c)$, konvex in $\hat c$ und nie kleiner als der SPO-Verlust; der Teilgradient ist $2\,(w^*(c) - w(2\hat c - c))$, braucht also zwei Löseraufrufe je Fall (auf $c$ und auf $2\hat c - c$).
- Normierter Regret: Summe der SPO-Verluste geteilt durch die Summe der besten Wegkosten (in Prozent).

Die Herleitung (Konvexität, obere Schranke, Teilgradient) steht im Expander „📐 Mathematische Formulierung“ der App.

## Methodik

- **Handbeispiel** (zwei Wege mit je einem Bogen, $c = (3, 5)$, $\hat c = (4, 3)$): gewählt wird Weg 2, SPO-Verlust 2, SPO+ 4, Teilgradient $(2, -2)$, Quadratfehler 5. Entlang $\hat c(t) = c + t\,(1, -2)$ ist SPO+ $= \max(0,\, 6t - 2)$.
- **Training.** Lineares Modell mit Achsenabschnitt (6 × 40 = 240 Parameter). KQ über `numpy.linalg.lstsq`. SPO+ mit **Adam**, Lernrate 0,05, Vollbatch, **300 Schritte**, Start bei der KQ-Lösung; zurückgegeben wird der Parameter mit dem kleinsten SPO+-Trainingsverlust unter allen besuchten (der Start gehört dazu, der Trainingsverlust des Ergebnisses ist also nie größer als der von KQ). Der Löser läuft in jeder Runde auf allen Trainingsfällen.
- **Auswertung.** Jede Zelle (Grad, n, Rauschstufe) läuft mit den festen Zufalls-Seeds 0 bis 19, 1000 Testfällen, 5 Merkmalen. Angegeben werden **Median und Spanne (Minimum bis Maximum)** über die Seeds, nie ein einzelner Seed. Die Bänder stehen vorgerechnet in `skw_bands.json` und sind mit `python tools/measure_bands.py` reproduzierbar (einige Minuten; Zufall nur über den ganzzahligen SplitMix64 aus `skw_rng.py`, nie Pythons `hash()`).
- **Zufallswald nicht live.** Die Zufallswald-Zahlen stammen aus einer **Vormessung mit scikit-learn** (100 Bäume, mindestens 5 Fälle je Blatt, nicht abgestimmt, 20 Seeds), eingefroren in `skw_wald_vormessung.json` (`tools/freeze_wald.py`); scikit-learn gehört nicht zu den Laufzeit-Paketen. In der App und hier heißt das Verfahren „Zufallswald (Vormessung, nicht live)“. Die Vormessung hat eine andere Zufallsquelle, ihre SPO+-Spalte benutzt 600 statt 300 Adam-Schritte: Zahlen der Vormessung und dieser Demo sind vergleichbar, nicht gleich.
- **Trainierte Zahlen sind plattformabhängig** (Adam, Gleitkomma, Löserentscheidungen bei knappen Unterschieden). Die Tests prüfen sie deshalb nur als Bänder und Ungleichungen; exakt geprüft werden nur Rechnungen ohne Training.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen dieses Abschnitts stammen aus `skw_bands.json` (eigene Messung: lineares Modell, KQ gegen SPO+ mit 300 Adam-Schritten, Zufalls-Seeds 0 bis 19, 1000 Testfälle, Regret in Prozent als Median [Minimum; Maximum]), wenn nichts anderes dabeisteht. Die Tabellen werden von `tests/readme_tables.py` aus der JSON-Datei erzeugt, und `tests/test_claims.py` prüft, dass jede Zeile wörtlich hier steht; die Aussagen darunter stehen als Bänder und Ungleichungen im selben Test.

### Raster ohne Rauschen (Rauschstufe 0)

| Grad | n | Orakel | KQ | SPO+ | SPO+ besser als KQ | Quadratfehler KQ / SPO+ |
|---|---|---|---|---|---|---|
| 1 | 100 | 0,0 | 0,0 [0,0; 0,0] | 0,0 [0,0; 0,1] | 0 von 20 | 0,00 / 0,00 |
| 1 | 1000 | 0,0 | 0,0 [0,0; 0,0] | 0,0 [0,0; 0,0] | 0 von 20 | 0,00 / 0,00 |
| 1 | 5000 | 0,0 | 0,0 [0,0; 0,0] | 0,0 [0,0; 0,0] | 0 von 20 | 0,00 / 0,00 |
| 2 | 100 | 0,0 | 0,1 [0,0; 0,1] | 0,2 [0,2; 0,3] | 0 von 20 | 0,00 / 0,30 |
| 2 | 1000 | 0,0 | 0,1 [0,0; 0,1] | 0,0 [0,0; 0,0] | 20 von 20 | 0,00 / 0,13 |
| 2 | 5000 | 0,0 | 0,1 [0,0; 0,1] | 0,0 [0,0; 0,0] | 20 von 20 | 0,00 / 0,03 |
| 4 | 100 | 0,0 | 0,8 [0,5; 1,5] | 0,6 [0,4; 0,8] | 18 von 20 | 0,11 / 0,67 |
| 4 | 1000 | 0,0 | 0,7 [0,4; 0,9] | 0,2 [0,1; 0,3] | 20 von 20 | 0,10 / 0,19 |
| 4 | 5000 | 0,0 | 0,7 [0,4; 0,9] | 0,2 [0,1; 0,2] | 20 von 20 | 0,10 / 0,16 |
| 6 | 100 | 0,0 | 2,6 [1,5; 5,1] | 1,2 [0,8; 2,3] | 20 von 20 | 0,82 / 2,36 |
| 6 | 1000 | 0,0 | 2,0 [1,4; 3,0] | 0,5 [0,4; 0,8] | 20 von 20 | 0,77 / 1,49 |
| 6 | 5000 | 0,0 | 2,0 [1,3; 3,1] | 0,5 [0,2; 0,7] | 20 von 20 | 0,76 / 1,39 |

### Raster mit Rauschen (Rauschstufe 0,5)

| Grad | n | Orakel | KQ | SPO+ | SPO+ besser als KQ | Quadratfehler KQ / SPO+ |
|---|---|---|---|---|---|---|
| 1 | 100 | 17,8 | 19,6 [18,6; 20,9] | 20,5 [18,9; 21,3] | 1 von 20 | 0,31 / 0,36 |
| 1 | 1000 | 17,8 | 18,1 [17,3; 18,9] | 18,3 [17,5; 19,3] | 6 von 20 | 0,29 / 0,30 |
| 1 | 5000 | 17,8 | 17,8 [17,2; 18,8] | 17,8 [17,1; 18,9] | 13 von 20 | 0,29 / 0,29 |
| 2 | 100 | 15,0 | 16,3 [14,9; 17,6] | 17,7 [15,9; 19,1] | 0 von 20 | 0,29 / 0,35 |
| 2 | 1000 | 15,0 | 15,4 [14,2; 16,2] | 15,4 [14,0; 16,4] | 6 von 20 | 0,28 / 0,28 |
| 2 | 5000 | 15,0 | 15,1 [14,0; 16,0] | 15,1 [14,2; 16,0] | 9 von 20 | 0,28 / 0,28 |
| 4 | 100 | 11,9 | 13,8 [11,7; 15,2] | 14,6 [13,1; 16,4] | 0 von 20 | 0,41 / 0,55 |
| 4 | 1000 | 11,9 | 12,8 [11,6; 14,1] | 12,7 [11,2; 13,8] | 13 von 20 | 0,39 / 0,39 |
| 4 | 5000 | 11,9 | 12,8 [11,3; 14,0] | 12,5 [11,3; 13,7] | 16 von 20 | 0,38 / 0,39 |
| 6 | 100 | 10,5 | 14,0 [11,5; 16,7] | 14,4 [12,5; 16,7] | 10 von 20 | 1,20 / 1,55 |
| 6 | 1000 | 10,5 | 13,1 [11,3; 14,5] | 12,3 [10,7; 13,4] | 20 von 20 | 1,15 / 1,19 |
| 6 | 5000 | 10,5 | 13,0 [10,9; 14,6] | 11,9 [10,5; 13,5] | 20 von 20 | 1,14 / 1,18 |

„SPO+ besser als KQ“ zählt die Seeds (von 20), in denen der Regret von SPO+ streng unter dem von KQ liegt. Der Quadratfehler ist der mittlere quadratische Fehler je Bogen auf den Testkosten (Median).

| Frage | Befund |
|---|---|
| **Bestes Regret, schlechteste Prognose?** | Ja, bei nichtlinearer Wahrheit: Grad 6, Rauschstufe 0, n = 5000: Regret KQ 2,0 %, SPO+ 0,5 % (SPO+ in 20 von 20 Seeds besser), Quadratfehler je Bogen KQ 0,76, SPO+ 1,39. In 12 Zellen der beiden Raster (8 ohne, 4 mit Rauschen), in denen der Median-Regret von SPO+ auf der angezeigten Stelle unter dem von KQ liegt, ist der Median-Quadratfehler von SPO+ nicht niedriger als der von KQ; ohne Rauschen ist er in allen 8 höher (zum Beispiel Grad 2, n = 1000: 0,00 gegen 0,13). In einer weiteren Zelle (Grad 1, n = 5000, mit Rauschen) liegen die Median-Regrets auf der angezeigten Stelle gleichauf (17,8 %); auch dort ist der Quadratfehler von SPO+ nicht niedriger. |
| **Wo gewinnt SPO+?** | Nicht bei linearer Wahrheit: Bei Grad 1 ist SPO+ in keiner Zelle ohne Rauschen besser (0 von 20 Seeds). Ohne Rauschen gewinnt SPO+ ab Grad 4 in allen Datenmengen (n = 100: Grad 4 in 18, Grad 6 in 20 von 20 Seeds; n = 1000 und 5000: 20 von 20). Mit Rauschen sicher nur bei Grad 6 und n = 1000 oder 5000 (je 20 von 20 Seeds); bei Grad 4 sind es 13 und 16 von 20, bei Grad 1 und 2 höchstens 13 von 20. |
| **Wie viel schließt SPO+ mit Rauschen?** | Grad 6, Rauschstufe 0,5, n = 1000: Orakel 10,5 %, KQ 13,1 %, SPO+ 12,3 %: SPO+ schließt 31 % der Lücke zwischen KQ und Orakel (aus den ungerundeten Medianwerten berechnet); bei n = 5000 (SPO+ 11,9 %) sind es 42 %. Der Rest ist unvermeidbares Rauschen (der Regret des Orakels) oder bleibt in der linearen Klasse. |
| **Verliert SPO+ bei wenig Daten?** | n = 100, Rauschstufe 0,5: bei Grad 1, 2 und 4 liegt SPO+ im Median über KQ (in 1, 0 und 0 von 20 Seeds besser; Differenz 1,1, 1,0 und 0,9 Punkte im Median der Paare). Bei Grad 6 ist es in diesen 20 Seeds ein Münzwurf (10 von 20); die Stabilitätsprobe (unten) mit 100 weiteren Seeds ergibt SPO+ besser in 29 von 100 und im Median 0,5 Punkte schlechter. 240 Parameter auf 100 Fälle. |
| **Hilft Strafe oder früher Abbruch?** | Mit Rauschen (n = 100, Grad 6) nicht erkennbar (Mediane, nicht auf Signifikanz geprüft): KQ 14,03 %, SPO+ ohne Strafe 14,39 %, mit λ = 0,1 14,34 %, mit λ = 1 14,21 %, nach 50 Schritten 14,16 %. Ohne Rauschen liegt λ = 0,1 knapp unter der Variante ohne Strafe (1,04 gegen 1,15 %, KQ 2,56 %), λ = 1 darüber (2,02 %). Die Zahl der Schritte ändert den Rückstand nicht (unten). |
| **Wie schnell trainiert SPO+?** | Grad 6, Rauschstufe 0, n = 1000: nach 25 Schritten sind im Median 99 % (Minimum 93 %, Maximum 104 %) des Regret-Gewinns zwischen KQ-Start und Schritt 300 erreicht. Mit Rauschen streut der Anteil stark (Minimum 74 %, Maximum 197 %) und sagt wenig; der Gewinn zwischen KQ-Start und Schritt 300 ist dort klein (Median 13,1 auf 12,2 %, Tabelle unten). Eine Aussage über Zeit ist das nicht: gemessen sind Adam-Schritte, keine Sekunden. |

### Lernkurve (Grad 6, n = 1000; Regret des Parameters nach t Adam-Schritten, Median [Minimum; Maximum] über 20 Seeds)

| Schritte | Regret ohne Rauschen (Median [Minimum; Maximum]) | Regret mit Rauschen 0,5 (Median [Minimum; Maximum]) |
|---|---|---|
| 0 | 2,0 [1,4; 3,0] % | 13,1 [11,3; 14,5] % |
| 5 | 0,7 [0,4; 0,9] % | 12,7 [10,5; 13,7] % |
| 10 | 0,6 [0,4; 0,9] % | 12,3 [10,9; 13,7] % |
| 25 | 0,5 [0,4; 0,9] % | 12,3 [10,5; 13,4] % |
| 50 | 0,5 [0,4; 0,8] % | 12,3 [10,7; 13,4] % |
| 100 | 0,5 [0,4; 0,9] % | 12,3 [10,6; 13,3] % |
| 300 | 0,5 [0,4; 0,8] % | 12,2 [10,7; 13,3] % |

### Strafe und früher Abbruch (Grad 6, n = 100, Median des Regrets über 20 Seeds)

| Variante | Rauschstufe 0 | Rauschstufe 0,5 |
|---|---|---|
| KQ | 2,56 % | 14,03 % |
| SPO+ ohne Strafe (300 Schritte) | 1,15 % | 14,39 % |
| SPO+ mit Strafe λ = 0,1 | 1,04 % | 14,34 % |
| SPO+ mit Strafe λ = 1 | 2,02 % | 14,21 % |
| SPO+ nach 50 Schritten | 1,12 % | 14,16 % |

### Zahl der Adam-Schritte (n = 100, Rauschstufe 0,5, Median des Regrets über 20 Seeds)

| Grad | Adam-Schritte | KQ | SPO+ | SPO+ besser als KQ |
|---|---|---|---|---|
| 4 | 300 | 13,8 % | 14,6 % | 0 von 20 |
| 4 | 600 | 13,8 % | 14,6 % | 0 von 20 |
| 4 | 1000 | 13,8 % | 14,6 % | 0 von 20 |
| 6 | 300 | 14,0 % | 14,4 % | 10 von 20 |
| 6 | 600 | 14,0 % | 14,6 % | 10 von 20 |
| 6 | 1000 | 14,0 % | 14,7 % | 9 von 20 |

### Stabilitätsprobe (n = 100, Rauschstufe 0,5, 100 weitere Zufalls-Seeds 20 bis 119)

| Grad | KQ (Median) | SPO+ (Median) | SPO+ besser als KQ | Median der Differenz SPO+ minus KQ |
|---|---|---|---|---|
| 4 | 13,6 % | 14,6 % | 6 von 100 | 0,94 Punkte |
| 6 | 13,7 % | 14,2 % | 29 von 100 | 0,53 Punkte |

### Zufallswald und SPO+ der Vormessung (Grad 6, nicht live, Zufalls-Seeds und Zufallsquelle der Vormessung)

Quelle: `entscheidungsorientiert-planung/vormessung/out_eo_v1_sp.txt` (Vormessung mit scikit-learn, 20 Seeds, 1000 Testfälle, SPO+ dort mit 600 Adam-Schritten), eingefroren in `skw_wald_vormessung.json`. Messung der Vormessung, nicht in dieser Demo gerechnet und nicht im Test nachgerechnet; der Test prüft nur, dass die Datei diese Zahlen enthält.

| n | Rauschstufe | Wald Regret | Wald Quadratfehler | KQ Quadratfehler | SPO+ (600 Schritte, Vormessung) Regret | SPO+ besser als KQ (Vormessung) |
|---|---|---|---|---|---|---|
| 100 | 0,0 | 5,4 % | 1,03 | 0,78 | 1,3 % | 19 von 20 |
| 1000 | 0,0 | 1,7 % | 0,61 | 0,76 | 0,6 % | 20 von 20 |
| 5000 | 0,0 | 0,8 % | 0,29 | 0,65 | 0,5 % | 20 von 20 |
| 100 | 0,5 | 17,2 % | 1,54 | 1,25 | 14,5 % | 2 von 20 |
| 1000 | 0,5 | 12,6 % | 0,88 | 1,05 | 11,8 % | 18 von 20 |
| 5000 | 0,5 | 11,5 % | 0,71 | 1,09 | 11,6 % | 19 von 20 |

Der Zufallswald ist unabgestimmt (100 Bäume, keine Wahl der Blattgröße); die Zahlen sind kein Urteil über Zufallswälder. Bei Grad 6, Rauschstufe 0,5 und n = 5000 sind Wald (11,5 %) und SPO+ (11,6 %) in der Vormessung gleichauf, KQ liegt bei 12,2 % (Rohausgabe).

### Prüfungen des Lösers und der Verluste

- **Löser gegen Aufzählung der 70 Wege:** 300 Instanzen mit stetigen Kosten, größte Abweichung im Wert 7·10⁻¹⁵, dieselben Wege; 300 Instanzen mit ganzzahligen Kosten (viele Gleichstände): dieselben Werte und dieselben Wege inklusive Gleichstandsregel. Scratch-Lauf, nicht im Repo; im Repo läuft die kleine Fassung (`tests/test_oracle_spo_kuerzester_weg.py`).
- **SPO+ gegen das Brute-Force-Maximum über alle 70 Wege:** 2000 Instanzen, größte Abweichung 1,4·10⁻¹⁴ (Scratch-Lauf).
- **Teilgradient gegen den zentralen Differenzenquotienten** (Richtungen ohne Knick): 2000 Instanzen, größte Abweichung 9,6·10⁻⁹ (Scratch-Lauf); die Teilgradienten-Ungleichung wird in 0 von 40000 Prüfungen verletzt, und SPO+ liegt in 0 von 2000 Fällen unter dem SPO-Verlust (Scratch-Lauf). Die Vormessung nannte 0 von 4000 Prüfungen.

## Befunde und Korrekturen gegenüber dem Plan

- **Die Zahlen der Vormessung reproduzieren sich nicht auf die Stelle.** Die Vormessung benutzte eine andere Zufallsquelle und 600 statt 300 Adam-Schritte (Entscheidung Z3: 300). Beispiel Grad 6, n = 1000: ohne Rauschen KQ 1,9 → 2,0 %, SPO+ 0,6 → 0,5 %; mit Rauschen KQ 12,5 → 13,1 %, SPO+ 11,8 → 12,3 %, SPO+ besser in 20 statt 18 von 20 Seeds. Es gilt die neue Messung. Die Rangfolge der Verfahren ist in diesen Zellen dieselbe.
- **„Bei n = 100 mit Rauschen verliert SPO+ gegen KQ“ (Plan: Grad 6, 2 von 20 Seeds besser) trägt für Grad 6 nur schwach.** In dieser Demo ist SPO+ bei Grad 6, n = 100, Rauschstufe 0,5 in 10 von 20 Seeds besser (Median 14,4 gegen 14,0 %). Eine Stabilitätsprobe mit 100 weiteren Seeds ergibt 29 von 100 und im Median 0,5 Punkte zu Lasten von SPO+; auch 600 oder 1000 statt 300 Schritte bringen die „2 von 20“ nicht zurück (10 und 9 von 20). Die Abweichung liegt demnach nicht an der Schrittzahl; die naheliegende Erklärung sind Seeds und Zufallsquelle (nicht weiter zerlegt). Deutlich schlechter als KQ ist SPO+ bei n = 100 mit Rauschen in den Welten mit Grad 1, 2 und 4 (1, 0 und 0 von 20 Seeds besser). Das Preset „Wenig Daten mit Rauschen“ benutzt deshalb Grad 4.
- **Strafe und früher Abbruch:** Die Aussage „helfen mit Rauschen nicht“ bleibt als „nicht erkennbar“ (Mediane über 20 Seeds: 14,16 bis 14,39 % für die SPO+-Varianten gegen 14,03 % für KQ; keine der Varianten hat einen Median unter KQ, nicht auf Signifikanz geprüft). Ohne Rauschen liegt λ = 0,1 mit 1,04 knapp unter 1,15 % (Unterschied klein, nicht auf Signifikanz geprüft) und damit weniger deutlich als in der Vormessung (1,02 gegen 1,34 %); λ = 1 liegt wie dort darüber.
- **Hinweis „SPO+ ist nach 25 Schritten fast am Endwert“** war in der Vormessung eine offene Messlücke; sie ist jetzt gemessen, mit der Einschränkung der Welt ohne Rauschen (Median 99 % des Gewinns, Spanne 93 bis 104 %).
- **Nur Zufallswald aus der Vormessung:** scikit-learn gehört nicht zu den Laufzeit-Paketen; ein numpy-Wald ist nicht gebaut. Die Zahlen des Waldes (Regret und Quadratfehler) sind damit auf den Zellen der Vormessung festgelegt und nicht live veränderbar.
- **Messung gegen die flexible Klasse nur als Hinweis:** Die Vormessung mit Gauß-Kern-Ridge (Grad 6, ohne Rauschen, n = 1000, 20 Seeds, `out_eo_v1d_kernel.txt`, SPO+ dort mit 300 Schritten, scikit-learn, nicht in dieser Demo) ergab Kern-Ridge 0,04 %, SPO+ 0,55 %, KQ 1,87 % im Median; mit Rauschen 0,5 (n = 1000) 11,33 % gegen 11,84 % (SPO+) und 12,47 % (KQ), bei n = 300 aber 12,50 % gegen 12,44 % (SPO+ knapp vorn) und bei n = 100 13,54 % gegen 14,35 % (SPO+), wobei KQ mit 13,43 % vorn liegt: ohne Rauschen klar, mit Rauschen weder klar noch durchgängig. Der Satz der Demo heißt darum „rettet das falsche lineare Modell“.

## Ehrliche Grenzen

- **Eine Welt, ein Gitter, ein Rauschmodell:** nur 5 × 5, nur diese Kostenformel (Grad 1, 2, 4, 6), Rauschstufe 0 und 0,5, ein lineares Modell, Adam mit fester Lernrate und 300 Schritten. Aussagen über andere Gitter, Wahrheiten, Trainer oder Lernraten sind nicht gemessen.
- **Das Gitter und die Kostenformel sind aus dem Gedächtnis nachgebaut**, nicht gegen den Primärtext geprüft. Elmachtoub und Grigas (Management Science 68(1), 2022) habe ich nur auf der Ebene des Abstracts gelesen; die Bedingungen ihrer Konsistenzaussage für SPO+ sind nicht im Primärtext geprüft und deshalb hier nicht wiedergegeben. Hu, Kallus und Mao (*Fast Rates for Contextual Linear Optimization*, Management Science 68(6), 2022, arXiv 2011.03030, Nummern nach arXiv v3, Beweise und Anhänge nicht gelesen) nennen SPO+ unter Regularitätsbedingungen Fisher-konsistent, wenn die wahre Bedingungserwartung in der Modellklasse liegt; ein Vergleich mit dem Gurobi-Training aus dieser Literatur fehlt.
- **Zahlen sind trainiert und plattformabhängig.** Gemessen unter Windows; unter Linux (CI und Streamlit Cloud) können einzelne Werte um Zehntelpunkte abweichen. Die Tests prüfen Bänder und Ungleichungen.
- **Median über 20 Seeds, kein Signifikanztest.** Die Spanne (Minimum bis Maximum) ist keine Konfidenzaussage; Zellen mit „10 von 20“ sind nicht unterscheidbar von gleichauf.
- **Die Rangfolge KQ gegen SPO+ gilt für das lineare Modell.** Gegen eine flexible Modellklasse sagt die Demo nichts Gemessenes in diesem Repo; die Vormessung (siehe oben) lässt SPO+ dort zurückliegen.
- **Ursachen sind Deutungen:** dass SPO+ bei wenig Daten verliert, weil 240 Parameter auf 100 Fälle kommen, ist die naheliegende Lesart, aber nicht durch einen eigenen Versuch (etwa mit weniger Merkmalen) belegt.
- **Keine Aussage über Rechenzeit.** Gemessen sind Adam-Schritte und Regret, keine Sekunden; die App nennt deshalb keine feste Zeit.

## Weitergabe: Module des Gitter-Gerüsts

Der Kern liegt in sechs Modulen ohne Streamlit-Abhängigkeit (nur numpy), die in anderen Demos kopiert werden können (nicht importiert):

| Modul | Inhalt |
|---|---|
| `skw_rng.py` | SplitMix64 (ganzzahlig), `derive_seed` für getrennte Teilströme |
| `skw_grid.py` | `Grid` (Löser als Rückwärtsrechnung, Gleichstandsregel, Aufzählung der Wege), `PathSet` (Löser über eine gegebene Wegmenge), `two_paths` (Handbeispiel) |
| `skw_world.py` | Kostenwelt: Merkmale, Gewichtsmatrix, mittlere und beobachtete Kosten, Trainings- und Testdaten |
| `skw_losses.py` | SPO-Verlust, SPO+ und Teilgradient, Quadratfehler, normierter Regret (für jeden Löser derselben Form) |
| `skw_training.py` | KQ, Vollbatch-Adam (`adam_minimize` für andere Gradientenfunktionen), SPO+-Training mit dem Löser in der Schleife |
| `skw_runs.py` | Lauf einer Welt (KQ, SPO+, Orakel, Lernkurve), Läufe über Seeds, Zusammenfassung |

## Verifikation

Die Suite hat **176 Tests**.

- **Kern gegen Handrechnung:** Löser auf 2 × 2, 3 × 3 und 5 × 5 mit von Hand gerechneten Instanzen (5 × 5: erste Zeile und letzte Spalte kosten 3, sonst 1: Wert 8, genau 20 optimale Wege, gewählt wird D R R R D D D R); das Handbeispiel (SPO 2, SPO+ 4, Teilgradient (2, −2), Quadratfehler 5), SPO+ entlang der Geraden $\max(0, 6t - 2)$; Adam-Schritte von Hand (`tests/test_grid.py`, `tests/test_losses.py`, `tests/test_training.py`).
- **Orakel** (`tests/test_oracle_spo_kuerzester_weg.py`, unabhängige Rechenwege): Löser gegen die Aufzählung aller 70 Wege (rekursiv erzeugt, eigene Summen, auch mit Gleichständen), SPO+ gegen das Brute-Force-Maximum über alle Wege, SPO gegen den Regret der Aufzählung, Teilgradient gegen den zentralen Differenzenquotienten, Teilgradient des Trainings gegen den Differenzenquotienten in den Parametern, KQ gegen die Normalengleichungen; SplitMix64 gegen eine reine Python-Ganzzahlrechnung (`tests/test_rng.py`).
- **Eigenschaften:** SPO+ ≥ SPO, SPO+ konvex, Teilgradienten-Ungleichung, SPO = 0 genau dann, wenn der gewählte Weg optimal ist, Skalieren von ĉ ändert den Weg nicht, Verschieben aller Bögen um eine Konstante ändert den Weg nicht, Trainingsverlust von SPO+ nie über dem von KQ.
- **Randfälle:** n = 1 und n = 2 Trainingsfälle, kleinstes Gitter, Nullkosten, Gleichstände, negative Vorhersagen, Randwerte der Regler (`tests/test_edge_cases.py`).
- **Claims:** jede Zahl dieser README steht in `tests/test_claims.py` als Band, Ungleichung oder wörtliche Tabellenzeile; die Vormessungswerte des Waldes werden nur als eingefroren geprüft.
- **Verdikte:** reine Funktionen, Schwellen und jede Zahl im Text (`tests/test_verdicts.py`); jedes Preset zeigt das erwartete Verdikt (`tests/test_app.py`).
- **Oberfläche:** AppTest (jeder Button, jeder Regler an den Grenzen, größte Welt, Permalink mit Einrasten), Layout (Schnellstart und Kostenwelt in der Seitenleiste, Regler der Experimente im Hauptbereich, fester Kopf, aktives Preset hervorgehoben), Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Seitenleiste (Schnellstart, Kostenwelt), Ergebnis, Kernfrage, fünf Experimente, Grenzen, Formeln |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, plotly, pandas, fpdf2 |
| `skw_rng.py`, `skw_grid.py`, `skw_world.py`, `skw_losses.py`, `skw_training.py`, `skw_runs.py` | Kern ohne Streamlit (siehe „Weitergabe“) |
| `skw_bands.py` | lädt die vorgerechneten Bänder |
| `skw_bands.json` | eigene Messung (`tools/measure_bands.py`) |
| `skw_wald_vormessung.json` | eingefrorene Zufallswald-Zahlen der Vormessung (`tools/freeze_wald.py`) |
| `skw_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `skw_evaluation.py` | Handbeispiel, Zahlen der Experimente, Verdikte (reine Funktionen), Tabellen aus den Bändern |
| `skw_layout.py` | Fester Kopf (Stylesheet) |
| `skw_presets.py` | Permalink (Begrenzen und Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `skw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `skw_pdf_export.py` | PDF-Export der aktuellen Welt (fpdf2) |
| `tools/measure_bands.py`, `tools/freeze_wald.py` | erzeugen die beiden JSON-Dateien |
| `tests/` | Tests (siehe „Verifikation“); `tests/readme_tables.py` erzeugt die Tabellen dieser README |

## Bewusst nicht umgesetzt

- **Zufallswald live:** braucht scikit-learn, das nicht zu den Laufzeit-Paketen gehört; die Zahlen sind eingefroren.
- **Flexible Modellklassen (Kern-Ridge, Wald) als Gegner von SPO+ in der App:** die Frage, wann welcher Weg gewinnt, ist hier nur als Hinweis auf die Vormessung enthalten.
- **Andere Gittergrößen, andere Wahrheiten, Minibatches, Lernratenwahl, Kreuzvalidierung der Strafe:** nicht gemessen.
- **Beweis der Konsistenz von SPO+ und Konvergenzraten:** nicht Gegenstand; die Demo zeigt den Mechanismus und Messungen.
- **Kürzeste-Wege-Algorithmen** (Dijkstra, Bellman-Ford): das Gitter ist azyklisch, der Löser eine Rückwärtsrechnung.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
