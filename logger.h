#pragma once
#include <stdio.h>
#include <stdarg.h>
#include <string>

class Logger{
private:
    const static std::string level_name[];
public:
    const static int ALL = 0;
    const static int DEBUG = 1;
    const static int INFO = 2;
    const static int WARN = 3;
    const static int ERROR = 4;

    static int log_level;

    static void log(int level, const char* format,...){
        if(level < log_level || level > 4 || level < 0)
            return;
        FILE* stream = stdout;
        if(level >= 3)
            stream = stderr;
        fprintf(stream,"[%s] ",level_name[level].c_str());
        va_list args;
        va_start(args,format);
        vfprintf(stream,format,args);
        va_end(args);
    }
};

const std::string Logger::level_name[] = {"ALL","DEBUG","INFO","WARN","ERROR"};
int Logger::log_level = Logger::ALL;
