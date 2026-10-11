#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__attribute__((used, noinline, noreturn)) void win(void) {
    char flag[512] = {0};
    int fd = open("/tmp/rlf-target-flag", O_RDONLY);
    if (fd < 0) _exit(2);
    ssize_t count = read(fd, flag, sizeof(flag) - 1);
    if (count > 0) write(STDOUT_FILENO, flag, (size_t)count);
    write(STDOUT_FILENO, "\n", 1);
    _exit(0);
}

__attribute__((noinline)) static void vulnerable(void) {
    char name[64];
    puts("name:");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    gets(name);
#pragma GCC diagnostic pop
    printf("hello %s\n", name);
}

int main(void) {
    setbuf(stdin, NULL);
    setbuf(stdout, NULL);
    alarm(20);
    vulnerable();
    return 0;
}

