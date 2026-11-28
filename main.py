from ejercicios.ejercicio1 import ejercicio1
from ejercicios.ejercicio2 import ejercicio2
from ejercicios.ejercicio3 import ejercicio3
from ejercicios.ejercicio4 import ejercicio4


def main():
    print("=== Reto 1: Conversion a escala de grises ===")
    print("Cierra las ventanas para continuar al Reto 2...")
    ejercicio1()

    print("\n=== Reto 2: Ajuste de contraste ===")
    print("Cierra las ventanas para continuar al Reto 3...")
    ejercicio2()

    print("\n=== Reto 3: Ecualizacion ===")
    print("Cierra las ventanas para continuar al Reto 4...")
    ejercicio3()

    print("\n=== Reto 4: Binarizacion ===")
    ejercicio4()


if __name__ == "__main__":
    main()
