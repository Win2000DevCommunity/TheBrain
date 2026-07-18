# TheBrain v13.0 - MSVC nmake
CC     = cl
CFLAGS = /TC /W3 /O2 /Gz /DMEM_BUDGET_MB=256 /D_WIN32_WINNT=0x0500 /DWINVER=0x0500
LIBS   = kernel32.lib user32.lib comctl32.lib comdlg32.lib shell32.lib \
         wininet.lib gdi32.lib advapi32.lib ole32.lib
LFLAGS = /link /SUBSYSTEM:WINDOWS

OBJS = sysinfo.obj tensor.obj graph.obj ops.obj ops_mt.obj model.obj \
       tokenizer.obj train.obj converse.obj brain_ml.obj brain_partA.obj brain_partB.obj

TheBrainV13.exe: $(OBJS)
	$(CC) $(OBJS) $(LIBS) $(LFLAGS) /OUT:TheBrainV13.exe

.c.obj:
	$(CC) $(CFLAGS) /c $< /Fo$@

clean:
	del /Q *.obj TheBrainV13.exe 2>NUL
