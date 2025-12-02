PROGNAME_V1 = assign3_2_v1
PROGNAME_V2 = assign3_2_v2
SRCFILES_V1 = assign3_2.c file.c timer.c simulate_v1.c
SRCFILES_V2 = assign3_2.c file.c timer.c simulate_v2.c
TARNAME = assign3_2.tgz

RUNARGS = 1000 1000 # i_max t_max, increase this when testing on the DAS4!
NODES = 1 # How many DAS4 nodes
PROCESSESPERNODE = 1 # How many processes to spawn on one machine.

PRUNARGS = -v -np $(NODES) -$(PROCESSESPERNODE) \
		   -sge-script $$PRUN_ETC/prun-openmpi
IMAGEVIEW = display
CC = mpicc

WARNFLAGS = -Wall -Werror-implicit-function-declaration -Wshadow \
		  -Wstrict-prototypes -pedantic-errors
CFLAGS = -std=c99 -ggdb -O2 $(WARNFLAGS) -D_POSIX_C_SOURCE=200112
LFLAGS = -lm -lrt

# Do some substitution to get a list of .o files from the given .c files.
OBJFILES_V1 = $(patsubst %.c,%.o,$(SRCFILES_V1))
OBJFILES_V2 = $(patsubst %.c,%.o,$(SRCFILES_V2))

.PHONY: all run runlocal plot clean dist todo v1 v2

all: $(PROGNAME_V1) $(PROGNAME_V2)

v1: $(PROGNAME_V1)

v2: $(PROGNAME_V2)

$(PROGNAME_V1): assign3_2.o file.o timer.o simulate_v1.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(PROGNAME_V2): assign3_2.o file.o timer.o simulate_v2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

run: $(PROGNAME_V1)
	prun $(PRUNARGS) $(PROGNAME_V1) $(RUNARGS)

run_v2: $(PROGNAME_V2)
	prun $(PRUNARGS) $(PROGNAME_V2) $(RUNARGS)

runlocal: $(PROGNAME_V1)
	mpirun -n $(PROCESSESPERNODE) $(PROGNAME_V1) $(RUNARGS)

runlocal_v2: $(PROGNAME_V2)
	mpirun -n $(PROCESSESPERNODE) $(PROGNAME_V2) $(RUNARGS)

plot: result.txt
	gnuplot plot.gnp
	$(IMAGEVIEW) plot.png

dist:
	tar cvzf $(TARNAME) Makefile *.c *.h

clean:
	rm -fv $(PROGNAME_V1) $(PROGNAME_V2) *.o $(TARNAME) result.txt plot.png
