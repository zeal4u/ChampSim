int data_array[10240] = {0};

int main(int argc, char const *argv[])
{
    /* code */
    for (int j = 0; j < 10000; j++)
    for (int i = 10240-2;i >= 0; i-=64) {
        data_array[i] = data_array[i+1] + 1;
    }
    return 0;
}
