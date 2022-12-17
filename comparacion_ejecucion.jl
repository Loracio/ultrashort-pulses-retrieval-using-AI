using FFTW
using Statistics


function pulso_gaussiano(t, A, τ, ω₀, φ)
    """
    Genera un pulso gaussiano dada su duración, frecuencia central y fase.
    Las unidades han de ser consistentes entre t, τ y ω_0.
    El pulso está normalizado.

    Un pulso gaussiano viene caracterizado por una envolvente en forma de gaussiana de expresión:

    E_envolvente = A * exp(-t² / 2*τ)

    Donde τ es la duración temporal del pulso, que está relacionada con el ancho de banda por la expresión:

    τ = FWHM / (2 * √log(2))

    FHWM es la anchura a media altura (full width half maximum).

    La envolvente viene modulada por un término exponencial complejo que depende de la frecuencia central de la onda,
    de manera que el pulso vendrá dado por el producto de la envolvente y esta exponeFourier Transform of Gaussian Modulated Functionncial, además del producto
    con la exponencial compleja que lleva la fase de la envolvente de la onda portadora.

    E(t) = E_envolvente * exp(i * ω_0 * t) * exp(i * φ(t)) = A * exp(-t² / 2*τ) * exp(i * ( ω_0 * t + φ(t) ) )

    Argumentos:
        t (float): vector de tiempos
        A (float): amplitud del pulso
        τ (float): anchura del pulso
        ω₀ (float): frecuencia central (radianes / unidad de tiempo)
        φ (float): fase de la envolvente de la onda portadora (rad)

    Devuelve:
        E_pulso (float): forma del pulso gaussiano en el tiempo especificado
    """
    return A .* exp.(-t.*t / (2 * τ)) .* exp.(1im * ( ω₀ * t .+ φ ))
end


function DFT(x::Vector{ComplexF64})
    """
    Implementación de la transformada discreta de Fourier (DFT).

    La transformada de Fourier viene dada por la siguiente integral:
        F(ω) = ∫f(x)e^{-i 2π ω x} dx

    Que en el caso de tener datos discretos se transforma en un sumatorio:
        Fₙ = ∑₀ᴺ⁻¹ fₖ e^{-i 2π k n / N}

    Cada Fₙ será el resultado de la transformada para el dato fₖ

    El número de operaciones requerido es del orden de O(N²), por lo que para grandes cantidades de datos no es muy eficiente.

    Args:
        x (Array{Complex{Float64}}): array de datos para obtener su transformada de Fourier

    Returns:
        (Array{Complex{Float64}}): array de datos con la transformada de los datos
    """
    n = length(x)

    W = zeros(Complex{Float64}, n, n)
    for i = 0:n-1
        for j = 0:n-1
            W[i+1, j+1] = exp(-im*2*pi*i*j/n)
        end
    end
    
    return W * x
end



function IDFT(x::Vector{ComplexF64})
    """
    Implementación de la transformada discreta de Fourier (DFT).

    La transformada de Fourier viene dada por la siguiente integral:
        F(ω) = ∫f(x)e^{-i 2π ω x} dx

    Que en el caso de tener datos discretos se transforma en un sumatorio:
        Fₙ = ∑₀ᴺ⁻¹ fₖ e^{-i 2π k n / N}

    Cada Fₙ será el resultado de la transformada para el dato fₖ

    El número de operaciones requerido es del orden de O(N²), por lo que para grandes cantidades de datos no es muy eficiente.

    Args:
        x (Array{Complex{Float64}}): array de datos para obtener su transformada de Fourier

    Returns:
        (Array{Complex{Float64}}): array de datos con la transformada de los datos
    """

    n = length(x)

    W = zeros(Complex{Float64}, n, n)
    for i = 0:n-1
        for j = 0:n-1
            W[i+1, j+1] = exp(im*2*pi*i*j/n)
        end
    end
    
    return W * x ./ n
end




function siguiente_potencia_dos(N::Int)
    """
    Devuelve la siguiente potencia de dos más cercana al número dado
    """
    return 2^(ceil(log2(N)))
end



function fft_propia(x::Vector{ComplexF64})
    """
    Implementación de la transformada rápida de Fourier (FFT) mediante el algoritmo de Cooley-Tukey.

    La transformada discreta de Fourier viene dada por:
        Fₙ = ∑₀ᴺ⁻¹ fₖ e^{-i 2π k n / N}

    Cada Fₙ será el resultado de la transformada para el dato fₖ

    El número de operaciones requerido es del orden de O(N²), por lo que puede llegar a ser computacionalmente costoso.

    El algoritmo de Cooley-Tukey es un algoritmo con una complejidad de O(n log n) que consigue reducir el número de operaciones
    dividiendo los datos de entrada en dos secuencias más pequeñas y calculando sus DFT recursivamente.

    Este algoritmo requiere que los datos de entrada tengan una longitud que sea una potencia de 2.
    En el caso que no se cumpla esto, puede rellenarse el array con ceros hasta la siguiente potencia de 2.
    El problema de esto es que la longitud del array devuelto será la de la longitud de la siguiente potencia de 2.
    Una solución para conservar la longitud del array origianal es usar el algoritmo de la transformada de Bluestein.

    Esta función realiza el rellenado del array si es necesario y llama recursivamente a una función
    que contiene el cuerpo del algoritmo.

    Args:
        x (Array{Complex}): array de datos para obtener su transformada de fourier inversa

    Devuelve:
        Array{Complex}: array de datos con la transformada inversa de los datos
    """
    N = length(x)

    if N != siguiente_potencia_dos(N)
        transformada_bluestein(x, 1)
    end

    return _fft_core(x, 1)
end




function ifft_propia(x::Vector{ComplexF64})
    """
    Implementación de la transformada rápida de Fourier inversa (IFFT).

    La transformada discreta de Fourier inversa viene dada por:
        fₖ = 1 / N · ∑₀ᴺ⁻¹ Fₙ e^{i 2π k n / N}

    Cada fₖ será el resultado de la transformada para el dato Fₙ

    El número de operaciones requerido es del orden de O(N²), por lo que puede llegar a ser computacionalmente costoso.

    El algoritmo de Cooley-Tukey es un algoritmo con una complejidad de O(n log n) que consigue reducir el número de operaciones
    dividiendo los datos de entrada en dos secuencias más pequeñas y calculando sus DFT recursivamente.

    Este algoritmo requiere que los datos de entrada tengan una longitud que sea una potencia de 2.
    En el caso que no se cumpla esto, puede rellenarse el array con ceros hasta la siguiente potencia de 2.
    El problema de esto es que la longitud del array devuelto será la de la longitud de la siguiente potencia de 2.
    Una solución para conservar la longitud del array origianal es usar el algoritmo de la transformada de Bluestein.

    Esta función realiza el rellenado del array si es necesario y llama recursivamente a una función
    que tiene el cuerpo del algoritmo.

    Finalmente, divide entre el número de muestras para dar el resultado final.

    Args:
        x (Array{Complex}): array de datos para obtener su transformada de fourier inversa

    Devuelve:
        Array{Complex}: array de datos con la transformada inversa de los datos
    """
    N = length(x)

    if N != siguiente_potencia_dos(N)
        return transformada_bluestein(x, -1) ./ N
    end

    return _fft_core(x, -1) ./ N
end




function _fft_core(x::Vector{ComplexF64}, signo::Int)
    """
    Cuerpo del algoritmo de la transformada rápida de Fourier.

    Esta función es llamada recursivamente para calcular la transformada de los índices pares
    e impares de los datos pasados como argumento.

    Una vez obtenidos se utilizan las ecuaciones del algoritmo de Cooley-Tukey que proporcionan
    los datos finales.

    Este algoritmo no es más que un algoritmo de multiplicación eficiente de polinomios.
    Más información sobre el funcionamiento del algoritmo en: https://www.youtube.com/watch?v=h7apO7q16V0

    Args:
        x (Array{Complex}): array de datos para obtener su transformada
        signo (float): signo del exponente de la transformada, que permite diferenciar en el caso de hacer la transformada o su inversa

    Devuelve:
        Array{Complex}: array de datos con la transformada de los datos
    """
    N = length(x)

    if N == 1
        return x
    end

    X_even = _fft_core(x[1:2:N], signo)
    X_odd = _fft_core(x[2:2:N], signo)

    X = Vector{ComplexF64}(undef, N)
    for k = 0:N÷2-1
        X[k+1] = X_even[k+1] + exp(- signo * 2π*im*k/N)*X_odd[k+1]
        X[k+1+N÷2] = X_even[k+1] - exp(- signo * 2π*im*k/N)*X_odd[k+1]

    end

    return X
end



function transformada_bluestein(x::Vector{ComplexF64}, signo::Int)
    """
    El algoritmo de Bluestein sirve para calcular la FFT de un array cuya longitud no es una potencia de 2. 
    El algoritmo de Bluestein consiste en los siguientes pasos:

        1) Rellenar la secuencia de entrada con ceros para hacer que tenga como longitud una potencia de 2.
        2) Multiplicar la secuencia rellenada por una secuencia especial de "chirp" para "desenrollar" la FFT.
        3) Calcular la FFT de la secuencia resultante utilizando el algoritmo Cooley-Tukey.
        4) Multiplicar la secuencia transformada por otra secuencia especial para "enrollar" de nuevo la DFT.

    Args:
        x (Array{Complex}): array de datos para obtener su transformada
        signo (float): signo del exponente de la transformada, que permite diferenciar en el caso de hacer la transformada o su inversa

    Devuelve:
        Array{Complex}: array de datos con la transformada de los datos
    """
    n = length(x)
    m = 2^(ndigits(n * 2, base = 2))
    
    coeficiente = - signo * pi / n
    exptable = exp.(1im * (collect(0:n - 1).^2 .% (n * 2)) * coeficiente) # Secuencia de chirp

    a = vcat(x .* exptable, zeros(m - n))

    b = vcat(exptable, zeros(m - (n * 2 - 1)), reverse(exptable)[1:(length(exptable) - 1)])
    
    b = conj(b)

    c = convolucion(a, b)[1:n]

    return c .* exptable
end



function convolucion(x::Vector{ComplexF64}, y::Vector{ComplexF64})
    """
    Calcula la convolución entre dos arrays de la misma longitud. 
    La convolución en el dominio temporal es igual al producto en
    el espacio de frecuencias, por lo que calculamos la fft de ambos
    arrays y luego multiplicamos sus elementos, para después calcular
    su transformada inversa.

    Args:
        x (Array{Complex}): array de datos para calcular su convolución con y
        y (Array{Complex}): array de datos para calcular su convolución con x

    Returns:
        Array{Complex}: convolución entre los dos arrays
    """
    n = length(x)

    x_transformada = fft_propia(x)
    y_transformada = fft_propia(y)

    z = ifft_propia(x_transformada .* y_transformada)

    return z
end

####################################################################################################################

numero_de_muestras = 2000

# -- Parámetros del pulso --
A = 1 # Amplitud del pulso
λ₀ = 1.55 # Longitud de onda de ejemplo (en micrómetros)
ω₀ = 2 * π * 2.99792458e8 * 1e-12 / (λ₀ * 1e-6) # Frecuencia angular del pulso (rad / ps)
φ₀ = fill(π / 4, numero_de_muestras) # Fase (constante en este caso)
τ = 1 # Duración del pulso (ps)

t = collect(range(-5, stop=5, length=numero_de_muestras))
pulso = pulso_gaussiano(t, A, τ, ω₀, φ₀)

n_repeticiones = 1000

# Compilacion separada. La primera llamada a la función hace que esta se compile.
DFT([1.0im])
fft([1.0im])
fft_propia([1.0im])


# Medimos tiempos de ejecucion de las distintas funciones
tiempos = Array{Float64}(undef,n_repeticiones)
tic = time()

for i in 1:n_repeticiones
    ela = @elapsed transformada = DFT(pulso)
    tiempos[i] = ela
end

toc = time()
tiempos *= 1e6

t   = mean(tiempos)
err = std(tiempos)

println("timing DFT: ", t," ± ", err," µs")



tiempos = Array{Float64}(undef,n_repeticiones)
tic = time()

for i in 1:n_repeticiones
    ela = @elapsed transformada = fft(pulso)
    tiempos[i] = ela
end

toc = time()
tiempos *= 1e6

t   = mean(tiempos)
err = std(tiempos)

println("timing FFTW: ", t," ± ", err," µs")



tiempos = Array{Float64}(undef,n_repeticiones)
tic = time()

for i in 1:n_repeticiones
    ela = @elapsed transformada = ifft_propia(pulso)
    tiempos[i] = ela
end

toc = time()
tiempos *= 1e6

t   = mean(tiempos)
err = std(tiempos)

println("timing propia: ", t," ± ", err," µs")

