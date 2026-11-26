# Run from the project root: gnuplot test/simplex_plot/plot.gp
set terminal svg size 1000,1000 font 'sans,12'
set datafile separator comma
set xlabel 't'
set xrange [0:1]
set grid back
set key opaque

out = 'test/output/simplex_plot/'
do for [precision in 'double dd_128 float128'] {
    bits = (precision eq 'double') ? 50 : 100
    dataset_count = (precision eq 'double') ? 5 : 3
    do for [dataset=1:dataset_count] {
        dim = dataset > 3 ? 3 : dataset
        suffix = dataset == 5 ? '_rank30_tci' : (dataset == 4 ? '_rank30' : '')
        title_suffix = dataset == 5 ? ', degree 29 (TCI fit)' : (dataset == 4 ? ', degree 29 (explicit TTN)' : '')
        stem = out . sprintf('%s_%dD%s', precision, dim, suffix)
        datafile = stem . '.csv'
        set output stem . '.svg'
        set multiplot layout 3,1 title sprintf('Simplex integration: %s, %dD, %d bits, 1000 samples%s', precision, dim, bits, title_suffix)

        unset logscale
        set xrange [0:1]
        set ylabel 'Q(t) (real part)'
        set key top left
        plot datafile using 1:2 with lines lw 3 lc rgb '#006eaa' title 'MPS', \
             '' using 1:3 with lines lw 2 dt 2 lc rgb '#e66100' title 'Discrete reference', \
             '' using 1:4 every 40 with points pt 6 lc rgb '#008658' title 'Continuous reference (every 40th point)'

        set logscale y
        set format y '%.0e'
        set ylabel 'Absolute error'
        set key bottom right
        plot datafile using 1:5 with lines lc rgb '#006eaa' title '|MPS - discrete|', \
             '' using 1:6 with lines lc rgb '#e66100' title '|MPS - continuous|', \
             '' using 1:9 with lines dt 2 lc rgb '#008658' title '|discrete - continuous|'

        set logscale x
        set xrange [0.001:1]
        set ylabel 'Relative error (t > 0)'
        plot datafile using 1:7 with lines lc rgb '#006eaa' title 'MPS vs. discrete', \
             '' using 1:8 with lines lc rgb '#e66100' title 'MPS vs. continuous'

        unset multiplot
        unset logscale
        set format y '%g'
        set output
    }
}
