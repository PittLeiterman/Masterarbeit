# Masterarbeit

## Konvergenzverhalten (Garantie und nicht konvexe Mengen)

Für konvexe Optimierungsprobleme besitzt ADMM unter milden Bedingungen eine globale Konvergenzgarantie – d.h. es konvergiert zum Optimalpunkt. Im vorliegenden (nichtkonvexen) Anwendungsfall gibt es diese Garantie theoretisch nicht unbedingt. Dennoch zeigen neuere Untersuchungen, dass ADMM auch in vielen nichtkonvexen Trajektorienplanungsproblemen konvergiert. Insbesondere wenn die Aufteilung so gestaltet ist, dass linear-affine (oder bikonvexe) Teilprobleme entstehen, kann man Konvergenzresultate ableiten.  
[Quelle](https://ar5iv.labs.arxiv.org/html/2111.07016)

---

## Vorteile von ADMM im Drohnenkontext

- **Dezentrale Aufteilung** – Jedes Subproblem lässt sich oft mit Standardmethoden sehr effizient lösen (z. B. QP-Solver für Quadratikziele, Projektionsoperator für Polyeder).
- **Parallelisierbarkeit** – Subprobleme weitgehend unabhängig.
- **Konvergenz zu feasiblen Lösungen** – ADMM kann so konzipiert werden, dass zu jedem Iterationszeitpunkt alle harten Constraints eingehalten werden (z. B. durch Start in feasiblem Korridor und geeignete Wahl der Penalitäten).
- **Schnelle Konvergenz in der Praxis** – Trotz fehlender Globalgarantie bei Nichtkonvexität zeigt ADMM exzellente praktische Performance.

---

## Herausforderungen bei der Anwendung von ADMM

- **Parametertuning** – ADMM erfordert mehr Feintuning als z.B. ein Interior-Point-Solver, der automatisch intern skaliert.
- **Konvergenzrate** – ADMM konvergiert zwar zuverlässig, aber nicht immer sehr schnell in Bezug auf Anzahl Iterationen. Es benötigt unter Umständen viele Iterationen, um hohe Genauigkeit zu erreichen (insbesondere bei streng gekoppelten Problemen).
- **Lokale Optimalität** – Wie die meisten Verfahren in nichtkonvexer Trajektorienoptimierung liefert ADMM keine Garantie auf das globale Optimum.

---

## Methodenvergleich

| Methode | Ansatz und Eigenschaften | Vorteile | Nachteile / Grenzen | Geeignet für |
|:---|:---|:---|:---|:---|
| **ADMM (Alternating Direction Method of Multipliers)** | Aufteilung in Teilprobleme mit Slack-Variablen und iterativer Abstimmung (Augmented Lagrangian). Löst bspw. getrennt Geometrie und Zeit, oder verteilt Segmente auf Worker. | - Parallelisierbar, skaliert gut auf große Probleme ([Quelle](https://ar5iv.labs.arxiv.org/html/2111.07016))<br>- Einhaltung von Constraints während Optimierung möglich<br>- Nutzt Speziallösungen (geschlossen/effizient) für Subprobleme<br>- Gute empirische Konvergenz mit hoher Geschwindigkeit | - Parameter ρ muss abgestimmt werden<br>- Ggf. viele Iterationen nötig (lokal linear konvergent)<br>- Nur lokale Optima garantiert (abhängig vom Initialpfad)<br>- Implementierung erfordert Problemzerlegung | Echtzeit-Optimierung großer Trajektorienprobleme; verteilte Berechnung; Multi-Drohnen-Koordination mit Kopplungen; Probleme mit vorhandener Initiallösung im zulässigen Bereich. |
| **MINCO (Minimum Control Effort mit Korridoren)** | Spezieller Optimierungs-Framework für polynomiale Trajektorien in SFCs. Wandelt Bahnplanung in ein glattes, unbeschränktes Optimierungsproblem um und löst es mit Gradientenverfahren (L-BFGS Quasi-Newton). | - Extrem schnell, Millisekundenlösung für Raum-Zeit-Optimierung<br>- Garantiert dynamisch glatte Trajektorien (Minimalkontrollaufwand)<br>- Nutzt analytische Gradienten; Open-Source Implementierungen verfügbar<br>- Speziell für aggressive Flüge entwickelt | - Im Prinzip auch lokales Verfahren<br>- Begrenzte Flexibilität: fokussiert auf polynomial parametrisierte Pfade<br>- Implementierung komplex<br>- Fixed-wing oder nicht-differentialflache Systeme schwer integrierbar | Hochgeschwindigkeitsflug mit vielen Segmenten; Onboard-Echtzeitplanung (z.B. Drohnenrennen, akrobatische Manöver). |
| **MIQP (Mixed-Integer Quadratic Programming)** | Formulierung als gemischt-ganzzahliges quadratisches Programm. Diskrete Variablen wählen z.B. Corridor-Regionen, kontinuierliche Variablen für Timing und Dynamik. | - Globale Optimalität möglich (im Rahmen der Diskretisierung)<br>- In einfachen Umgebungen gute Rechenzeit<br>- Klar formuliertes Korridor-Constraint-Modell | - Kombinatorische Explosion bei vielen Alternativen<br>- Lösungsgüte hängt von Diskretisierung ab<br>- Starre Lösungen; Glattheit schwer einbringbar | Offline-Planung oder kleine Räume mit wenigen Wegwahlmöglichkeiten; Routenplanung auf hoher Ebene. |
| **SQP (Sequential Quadratic Programming)** | Nichtlineare Optimierung mittels sequentieller QP-Schritte. Löst iterativ approximierte QP-Unterschritte des vollständigen Problems. | - Generisch und flexibel<br>- Hohe Genauigkeit erreichbar<br>- Gut handhabbar bei moderaten Problemgrößen | - Rechenaufwendig bei vielen Variablen<br>- Braucht gute Initiallösung<br>- Kann während Iterationen ungültige Trajektorien erzeugen<br>- Nicht trivial parallelisierbar | Spezialfälle mit wenigen Segmenten; Fine-Tuning einer bestehenden Trajektorie; Offline-Optimierung bei nicht zeitsensiblen Anwendungen. |
| **SCP (Sequential Convex Programming)** | Iteratives Konvexifizierungsverfahren, z.B. CHOMP oder TrajOpt. Optimiert schrittweise durch lokale Konvexifikation nichtlinearer Bestandteile. | - Vereint Sicherheit und Optimierung<br>- Handhabung anspruchsvoller Constraints<br>- Oft wenige Iterationen nötig | - Nur lokal optimal, abhängig vom Initialpfad<br>- Schwierigkeit, sehr unterschiedliche Lösungen zu finden<br>- Zeitaufwendig pro Iteration | Probleme mit gegebener heuristischer Pfadlösung; geeignet für Umgebungen mit Hindernissen und anschließende Pfadverfeinerung. |

---


## nächste Schritte
- bessere Methode, um die Startpunkte zu initialisieren
- Iteratives Verfahren aktivieren, nicht nur einfache Optimierung


## Problem mit reinen Ansätzen

| Ansatz            | Schwäche |
|-------------------|----------|
| **Nur Wu (Seed-basiert)** | Seed-Update ist oft heuristisch, Optimierung nicht modular, schwer zu parallelisieren |
| **Nur Ni (ADMM)**         | Keine Kontrolle über Raumstruktur, angewiesen auf gute initiale Korridore oder Separating Planes |

---

## Was macht ein Hybridansatz?

### Ablauf (vereinfacht):

1. **Initialer Pfad** (z. B. RRT*)
   - Generiert erste Seedpunkte
2. **Initiale SFC-Erstellung** um diese Seeds  
   *(z. B. Ellipsoide, IRIS, CIRI)*
3. **ADMM-Loop startet**:
   - **Subproblem A**: Optimierung der Trajektorie *(wie in Ni et al.)*
   - **Subproblem B**: Seedpunkt-Anpassung + neue Korridore *(wie in Wu et al.)*
   - **Synchronisation** über *Slack-Variablen*:
     - Positionen  
     - Volumen  
     - Dynamikgrenzen
4. **Wiederhole**, bis Konvergenz erreicht ist




## Quellen
- [ADMM Convergence for Nonconvex Problems (ar5iv.org)](https://ar5iv.labs.arxiv.org/html/2111.07016)
- Weitere Quellen im Text verlinkt.
