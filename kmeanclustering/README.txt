Kurulum ve ayarlama

1.) Scriplerin çalýþmasý için
    -opencv
    -scikit-image
    -numpy
    modülleri eklenmelidir. scikit-image python 32 bit sürümü ile çalýþmadýðýndan python 64 bit sürümü kullanýlmasý þarttýr. Ayrýca scikit-image modülü kimi platformlarda derleme gerekitirebileceðinden Windows platformu üzerinde visual c++ build tools kurulmasýnda fayda vardýr. Ayrýca environment variables patikalarýna da C:\Program Files (x86)\Windows Kits\8.1\bin\x86 patikasýnýn eklenmesi gerekmektedir.

2.) Dataset resimlerinin eðitim için train ve test için dataset klasörlerinin içerisine konmasý gereklidir. Her bir class bir klasör haline
getirilip train ve dataset klasörlerinin içine konmalýdýr. Örnek olmasý açýsýndan göderilen klasörün içerisinde 1 tane resim ve klasör yapýsý býrakýlmýþtýr.
    Ör : dataset/patates/
	 dataset/karpuz/
         gibi..

3.) main.py dosyasý içerisinde

    training = np.loadtxt("out.txt")
    #training = filefinder.getTrainingData()

methodlarý bulunmaktadýr. Bu methodlardan ayný anda sadece birisi çalýþacak þekilde tasarlanmýþtýr.  Eðer birinci method çalýþtýrýlýr ise 
daha önceden train modu çalýþtýrýp oluþturulmuþ out.txt dosyasýndan verileri okur ve classification bu dataya göre yapýlýr. Eðer ikinci satýrdaki method çalýþýtýrýlýr ise program çalýþtýrýldýðýnda eðitim moduna geçmektedir ve sonuçlarý programýn çalýþtýrýldýðý klasörde out.txt adlý dosyaya kaydetmektedir. Feature extraction uzun süren iþlem olduðundan bir kez train modu çalýþtýrýldýktan sonra sonraki denemelerde out.txt dosyasýndan kullanýlmasýný öneririm