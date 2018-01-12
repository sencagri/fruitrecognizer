Kurulum ve ayarlama

1.) Scriplerin �al��mas� i�in
    -opencv
    -scikit-image
    -numpy
    mod�lleri eklenmelidir. scikit-image python 32 bit s�r�m� ile �al��mad���ndan python 64 bit s�r�m� kullan�lmas� �artt�r. Ayr�ca scikit-image mod�l� kimi platformlarda derleme gerekitirebilece�inden Windows platformu �zerinde visual c++ build tools kurulmas�nda fayda vard�r. Ayr�ca environment variables patikalar�na da C:\Program Files (x86)\Windows Kits\8.1\bin\x86 patikas�n�n eklenmesi gerekmektedir.

2.) Dataset resimlerinin e�itim i�in train ve test i�in dataset klas�rlerinin i�erisine konmas� gereklidir. Her bir class bir klas�r haline
getirilip train ve dataset klas�rlerinin i�ine konmal�d�r. �rnek olmas� a��s�ndan g�derilen klas�r�n i�erisinde 1 tane resim ve klas�r yap�s� b�rak�lm��t�r.
    �r : dataset/patates/
	 dataset/karpuz/
         gibi..

3.) main.py dosyas� i�erisinde

    training = np.loadtxt("out.txt")
    #training = filefinder.getTrainingData()

methodlar� bulunmaktad�r. Bu methodlardan ayn� anda sadece birisi �al��acak �ekilde tasarlanm��t�r.  E�er birinci method �al��t�r�l�r ise 
daha �nceden train modu �al��t�r�p olu�turulmu� out.txt dosyas�ndan verileri okur ve classification bu dataya g�re yap�l�r. E�er ikinci sat�rdaki method �al���t�r�l�r ise program �al��t�r�ld���nda e�itim moduna ge�mektedir ve sonu�lar� program�n �al��t�r�ld��� klas�rde out.txt adl� dosyaya kaydetmektedir. Feature extraction uzun s�ren i�lem oldu�undan bir kez train modu �al��t�r�ld�ktan sonra sonraki denemelerde out.txt dosyas�ndan kullan�lmas�n� �neririm