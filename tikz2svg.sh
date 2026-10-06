#!/usr/bin/env bash
set -e

# Abbruch bei fehlendem Argument
if [ -z "$1" ]; then
    echo "Verwendung: $0 <eingabe.tex> [breite, z.B. 20cm] [ausgabe.svg]"
    echo "Beispiel:   $0 tikz/methods/overall_training_procedure.tex"
    echo "Beispiel:   $0 tikz/methods/overall_training_procedure.tex 22cm custom_name.svg"
    exit 1
fi

INPUT_FILE="$1"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Fehler: Datei '$INPUT_FILE' existiert nicht!"
    exit 1
fi

# Standardwerte festlegen
WIDTH="20cm"
OUTPUT_SVG=""

# Parameter 2 & 3 intelligent auflösen (Breite vs. .svg Dateiname)
if [ -n "$2" ]; then
    if [[ "$2" == *.svg ]]; then
        OUTPUT_SVG="$2"
    else
        WIDTH="$2"
    fi
fi

if [ -n "$3" ]; then
    OUTPUT_SVG="$3"
fi

# Falls keine Ausgabedatei angegeben: gleicher Ordner/Name wie Eingabedatei (.svg)
if [ -z "$OUTPUT_SVG" ]; then
    DIR=$(dirname "$INPUT_FILE")
    BASE=$(basename "$INPUT_FILE" .tex)
    OUTPUT_SVG="${DIR}/${BASE}.svg"
fi

JOBNAME="temp_tikz_export"

echo "==> Kompiliere '$INPUT_FILE' mit \\linewidth = $WIDTH..."
if ! lualatex -interaction=nonstopmode --jobname="$JOBNAME" \
    "\def\customwidth{$WIDTH}\def\tikzfile{$INPUT_FILE}\input{tikz_helper.tex}" > /dev/null 2>&1; then
    if [ ! -f "${JOBNAME}.pdf" ]; then
        echo "Fehler: Kompilierung fehlgeschlagen! Bitte '${JOBNAME}.log' prüfen."
        exit 1
    fi
fi

echo "==> Konvertiere zu SVG via pdftocairo..."
pdftocairo -svg "${JOBNAME}.pdf" "$OUTPUT_SVG"

# Temporäre Hilfsdateien bereinigen
rm -f "${JOBNAME}".{aux,log,pdf}

echo "✓ Erfolgreich erstellt: $OUTPUT_SVG"