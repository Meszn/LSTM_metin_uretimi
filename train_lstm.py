"""
Problem tanimi: LSTM ile metin uretimesi yapilacak. Verilen kelimelerden anlamli cumleler uretilmeye calisacak.
    -ben yarin ... gidecegim

LSTM:Long Short-Term Memory, RNN'lerin bir türüdür ve uzun vadeli bağımlılıkları öğrenmek için tasarlanmıştır. LSTM'ler, geleneksel RNN'lere göre daha iyi performans gösterirler çünkü unutma kapıları ve giriş kapıları gibi mekanizmalar kullanarak bilgiyi daha etkili bir şekilde saklarlar.

Veri Seti: Chatgpt ile üretilmiş 100 cümlelik bir veri seti kullanılacak. Bu cümleler, çeşitli konularda ve yapısal olarak farklı olabilir.
Model Eğitimi: LSTM modeli, bu veri seti üzerinde eğitilecek. Model, verilen kelimelerden anlamlı cümleler üretmeyi öğrenecek.



"""
#import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# egitim verisini chatgpt ile olustur 
data = [
    "Bugün hava top oynamak için çok güzel.",
    "Yarın sabah erken kalkmam gerekiyor.",
    "Akşam yemeğinde ne yiyeceğimizi henüz bilmiyorum.",
    "Marketten gelirken iki ekmek alır mısın?",
    "Hafta sonu arkadaşlarımla pikniğe gideceğiz.",
    "Bu kitabı okumayı gerçekten çok seviyorum.",
    "Bugün iş yerinde çok yoğun bir gün geçirdim.",
    "En sevdiğim yemek patlıcan musakkadır.",
    "Kedim bütün gün koltuğun üzerinde uyuyor.",
    "Akşamki maçı izlemek için sabırsızlanıyorum.",
    "Arabanın anahtarını nerede bıraktığımı hatırlamıyorum.",
    "Yeni açılan restorana gitmeyi planlıyoruz.",
    "Çay mı içersin yoksa kahve mi istersin?",
    "Telefonumun şarjı bitmek üzere, hemen prize takmalıyım.",
    "Dışarı çıkarken üzerine ince bir hırka al.",
    "Bayramda ailemi ziyaret etmek beni çok mutlu ediyor.",
    "Televizyonda çok ilginç bir belgesel başlıyor.",
    "Balkondaki çiçekleri sulamayı sakın unutma.",
    "Bu sabah kahvaltı yapmaya hiç vaktim olmadı.",
    "Bilgisayarım bugün nedense çok yavaş çalışıyor.",
    "Düzenli spor yapmak insanı gerçekten zinde tutuyor.",
    "Yaz tatilinde Ege kıyılarına gitmeyi çok istiyorum.",
    "Yarınki sınav için gece boyu çalışmam lazım.",
    "Otobüs durağında on beş dakikadır bekliyorum.",
    "Akşam yemeğinden sonra biraz yürüyüş yapalım mı?",
    "Bugün kendimi biraz yorgun ve halsiz hissediyorum.",
    "Sinemadaki yeni filmi izlemeyi çok merak ediyorum.",
    "Mutfaktaki bulaşıkları makineye yerleştirmem gerekiyor.",
    "Okuldan sonra kütüphaneye gidip ders çalışacağım.",
    "En sevdiğim renk aslında bebemavisidir.",
    "Bu sokaktaki fırının simitleri her zaman taze oluyor.",
    "Tatil planları yapmak beni her zaman heyecanlandırır.",
    "Evin önündeki ağaç baharın gelmesiyle çiçek açmış.",
    "Bugün acilen alışveriş merkezine gitmem lazım.",
    "Arkadaşımın doğum günü için güzel bir hediye bakıyorum.",
    "Yatmadan önce kitap okumak beni çok rahatlatıyor.",
    "Bugün trafikte beklerken çok fazla vakit kaybettim.",
    "Akşam bize gel de birlikte güzel bir çay içelim.",
    "Pencereyi kapatır mısın, içerisi aniden soğudu.",
    "Ödevlerimi bitirdikten sonra biraz bilgisayar oyunu oynayacağım.",
    "Haberlerde yarın havanın yağmurlu olacağını söylediler.",
    "Bugün annemle birlikte çok lezzetli bir kek yaptık.",
    "Şehrin gürültüsünden bazen gerçekten çok sıkılıyorum.",
    "Eski fotoğraflara bakmak beni hep çocukluğuma götürür.",
    "Her sabah yürüyüş yapmak fiziksel sağlık için çok yararlı.",
    "Hafta sonu evi baştan aşağı temizlemem gerekiyor.",
    "Kitap okurken zamanın nasıl geçtiğini hiç anlamıyorum.",
    "Bu akşam televizyonda en sevdiğim dizi yayınlanacak.",
    "Çantamda anahtarımı bulamayınca bir an çok panikledim.",
    "Yeni bir dil öğrenmeye çalışmak gerçekten çok keyifli."
    "Yeni aldığım ayakkabılar ayağımı biraz vurdu.",
    "Akşam dışarı çıkarken kapıyı kilitlemeyi sakın unutma.",
    "Bu hafta sonu için sinemadan iki kişilik bilet aldım.",
    "Annemle telefonda bugün uzun uzun dertleştik.",
    "Bilgisayarın ekranı aniden karardı, ne olduğunu anlamadım.",
    "Kahve makinesini temizlemek gerçekten çok zahmetliymiş.",
    "Yarınki büyük toplantı için sunum hazırlamam gerekiyor.",
    "Bahçedeki kediler için marketten mama almam lazım.",
    "Bu sabah uyandığımda kendimi çok dinç hissettim.",
    "Akşamki yemek davetine ne giyeceğime bir türlü karar veremedim.",
    "Parkta yürürken eski bir ilkokul arkadaşıma rastladım.",
    "Marketin önündeki kuyruk bugün inanılmaz derecede uzundu.",
    "Yaz mevsiminin gelmesini ve deniz tatilini dört gözle bekliyorum.",
    "Akşam yemeğinde taze fasulye ve yanına pilav var.",
    "Bugün işten eve dönerken radyoda çok güzel bir şarkı dinledim.",
    "Evin içindeki eşyaların yerini değiştirmeyi düşünüyorum.",
    "Yeni bir hobi edinmek için internette kurs araştırması yapıyorum.",
    "Çay bardağını masanın kenarına koyma, her an düşebilir.",
    "Hafta sonu ailemle birlikte kahvaltı yapmayı çok seviyorum.",
    "Bu akşam erkenden uyuyup iyice dinlenmek istiyorum.",
    "Arkadaşımın düğünü için şık bir elbise arıyorum.",
    "Televizyonun sesini biraz kısar mısın, içeride ders çalışıyorum.",
    "Bugün hava dünküne göre çok daha ılık ve güneşli görünüyor.",
    "Okuduğum kitabın sonu beni gerçekten çok şaşırttı.",
    "Mutfak dolaplarını düzenlemek için bütün gün uğraştım.",
    "En sevdiğim meyve olan çileğin mevsimi sonunda geldi.",
    "Yarın sabah erkenden spor salonuna gitmeyi planlıyorum.",
    "Otobüsün gelmesine daha on dakika varmış gibi görünüyor.",
    "Bugün ofiste bir arkadaşımızın doğum günü için pasta kestik.",
    "Dışarıdaki rüzgarın sesi evin içine kadar geliyor.",
    "Telefonumun kamerasının bozulduğunu fark edince çok üzüldüm.",
    "Akşam yemeğinden sonra güzel bir meyve tabağı hazırlayacağım.",
    "Çantamın içini tamamen boşaltıp yeniden düzenlemem lazım.",
    "Yeni aldığım kulaklığın ses kalitesi gerçekten harika.",
    "Arkadaşımla sahilde kahve içip bol bol sohbet ettik.",
    "Bugün öğle uykusuna daldığım için gece bir türlü uyuyamadım.",
    "Kardeşimin matematik ödevlerine yardım etmek bazen çok zor oluyor.",
    "Balkonda oturup gökyüzünü izlemek beni huzurlu hissettiriyor.",
    "Yarınki okul gezisi için sırt çantamı şimdiden hazırladım.",
    "Evdeki ampullerden biri patladığı için oda karanlık kaldı.",
    "En sevdiğim şarkıcı önümüzdeki ay bizim şehre geliyormuş.",
    "Çiçeklerin saksısını değiştirmek için çiçekçiden yeni toprak aldım.",
    "Bugün yolda yürürken çok sevimli bir köpek gördüm.",
    "Akşam yemeği için dışarıdan pizza sipariş vermeye karar verdik.",
    "Bilgisayar oyunları oynamak günün stresini atmama yardımcı oluyor.",
    "Odadaki perdeleri yıkamak için sabah erkenden yerinden çıkardım.",
    "Bugün spor yaparken kendimi biraz fazla zorlamışım, her yerim ağrıyor.",
    "Akşamki film izleme keyfi için bolca mısır patlatacağım.",
    "Yeni bir araba almak için şimdiden para biriktirmeye başladım.",
    "Güne taze sıkılmış portakal suyu ile başlamak harika hissettiriyor."
    "Sabah alarmı duymadan uyanmışım, kendimi çok dinç hissediyorum.",
    "Bugün hava gerçekten çok soğuk, dışarı çıkarken kalın giyinmek lazım.",
    "Kahvaltıda taze demlenmiş bir çay gibisi yok.",
    "Kampüse yürürken yolda sevdiğim bir şarkıyı dinledim.",
    "İlk derse yetişmek için adımlarımı biraz hızlandırmam gerekti.",
    "Kantin sırası o kadar uzundu ki kahve almaktan vazgeçtim.",
    "Öğle arasında arkadaşlarımla buluşup yemek yiyeceğiz.",
    "Bugün kütüphanede sessiz bir köşe bulup biraz çalışmam lazım.",
    "Hocanın verdiği son ödev sandığımdan daha oyalayıcı çıktı.",
    "Vize haftası yaklaştığı için stres seviyem yavaş yavaş artıyor.",
    "Notlarımı temize çekmek akşamımı alacak gibi duruyor.",
    "Ders çıkışı biraz hava almak için kampüste yürüyüş yaptım.",
    "Python kodundaki o küçük hatayı bulmak saatlerimi aldı.",
    "Modelin eğitim süresi beklediğimden çok daha uzun sürdü.",
    "Veri setini temizlemek her zaman projenin en sıkıcı kısmı oluyor.",
    "Log kayıtlarındaki anormallikleri tespit etmek için yeni bir script yazdım.",
    "Algoritmanın doğruluğunu artırmak için farklı parametreler deniyorum.",
    "Github'a son değişiklikleri pushlamayı unuttuğumu fark ettim.",
    "Takım toplantısında yeni algoritmaları ve stratejileri tartıştık.",
    "Staj raporumu yarına kadar sisteme yüklemem gerekiyor.",
    "Sensörlerden gelen verileri işlemek için yeni bir kütüphane kurdum.",
    "Ekrandaki hatayı çözünce derin bir oh çektim.",
    "Gözlemevinde çalışmak gökyüzüne bakış açımı tamamen değiştirdi.",
    "Mentörlük yaptığım öğrencilerle bugün harika bir ders işledik.",
    "Projeyi zamanında yetiştirebilmek için hafta sonu da çalışmalıyım.",
    "Bilgisayarın başında çok oturmaktan sırtım ağrımaya başladı.",
    "Kod yazarken arkada enstrümantal müzik çalması odaklanmamı sağlıyor.",
    "Yazılım dökümantasyonunu okumak bazen kod yazmaktan daha zor.",
    "Uygulamanın arayüzünü biraz daha kullanıcı dostu yapmalıyız.",
    "Server çöktüğü için bütün işlemlere baştan başlamak zorunda kaldık.",
    "Annemle telefonda konuşmak bana her zaman çok iyi gelir.",
    "Kız arkadaşımla hafta sonu için çok güzel bir plan yaptık.",
    "Annem eve gelirken gelirken ekmek almamı tembihledi.",
    "Kız arkadaşımın rehberde kayıtlı ismini değiştirmeyi düşünüyorum.",
    "Ailemle akşam yemeğinde bir araya gelmeyi çok özledim.",
    "Bugün ev arkadaşımla mutfak alışverişine çıkmamız gerekiyor.",
    "Kardeşim bilgisayarımı bozduğu için ona biraz sinirlendim.",
    "Eski mahallemden bir arkadaşımla yolda karşılaştım.",
    "Misafirler gelmeden önce salonu hızlıca bir toparladım.",
    "Babama yeni telefonunu kurmasında yardımcı oldum.",
    "Kız arkadaşıma doğum günü için sürpriz bir hediye aldım.",
    "Bugün nedense içimde sebepsiz bir huzur var.",
    "Dün gece izlediğim filmin etkisinden hala çıkamadım.",
    "Sürekli aynı şeyleri yapmaktan bazen çok sıkılıyorum.",
    "Yeni bir şeyler öğrenmek beni her zaman motive eder.",
    "Bazen sadece sessizce oturup uzaklara bakmak istiyorum.",
    "Trafikteki kornalar yüzünden başıma ağrı girdi.",
    "Sokak hayvanları için kapının önüne bir kap su koydum.",
    "Gökyüzündeki bulutların şekli bugün çok ilginç görünüyor.",
    "En sevdiğim kafede oturup camdan insanları izledim.",
    "Cüzdanımı evde unuttuğumu kasaya gelince fark ettim.",
    "Market poşetleri o kadar ağırdı ki ellerim koptu.",
    "Yeni aldığım kitabın ilk elli sayfasını bir çırpıda okudum.",
    "İnternet bağlantısı sürekli koptuğu için sinirlerim bozuldu.",
    "Ayakkabımın bağcığı yolda yürürken üç kez çözüldü.",
    "Güneşli havalarda insanın enerjisi gerçekten yerine geliyor.",
    "Yağmur yağarken evde oturup film izlemeye bayılıyorum.",
    "Şarj kablom koptuğu için yenisini sipariş ettim.",
    "Bu akşam yemek yapmaya hiç üşendiğim için dışarıdan söyledik.",
    "Televizyonda izleyecek hiçbir şey bulamayınca kapattım.",
    "Bugün o kadar çok yürüdüm ki ayaklarımı hissetmiyorum.",
    "Dolapta yiyecek hiçbir şey kalmamış, alışveriş şart.",
    "Komşunun köpeği bütün gece havladığı için uyuyamadım.",
    "Saçlarımı kestirmenin vakti geldi de geçiyor bile.",
    "Sabah evden çıkarken anahtarı alıp almadığımı üç kez kontrol ettim.",
    "Asansör bozuk olduğu için altıncı kata kadar merdiven çıktım.",
    "Otobüste yanıma oturan teyze yol boyunca hiç susmadı.",
    "Gözlüklerimin camı sürekli lekelendiği için temizlemekten yoruldum.",
    "Bugün nedense sürekli tatlı bir şeyler yemek istiyorum.",
    "Çaydanlığı ocakta unuttuğum için mutfağı duman kapladı.",
    "Alarmı beş dakika daha ertelemek en büyük hobim oldu.",
    "Duş aldıktan sonra kendime kocaman bir kupa kahve yaptım.",
    "Yeni aldığım tişört ilk yıkamada hemen çekti.",
    "Bugün hiçbir şey yapmadan sadece yatmak istiyorum.",
    "Yarınki hava durumuna bakıp ona göre plan yapacağım.",
    "Kulaklığımın tek tarafından ses gelmemeye başladı.",
    "Gece yarısı acıkınca mutfakta gizlice atıştırmalık aradım.",
    "Sosyal medyada gezinirken zamanın nasıl geçtiğini anlamıyorum.",
    "Camdan dışarı bakıp geçen arabaları saydım.",
    "Bugün çok fazla su içmediğimi fark edip hemen bir bardak doldurdum.",
    "Odamı havalandırmak için bütün pencereleri sonuna kadar açtım.",
    "Kargocu evde olmadığım için paketi şubeye götürmüş.",
    "Yeni aldığım defterin yapraklarının kokusu çok güzel.",
    "Bugün nedense sürekli eski anılar aklıma geliyor.",
    "Tırnaklarımı keserken bir tanesini çok derinden kestim.",
    "Aynaya bakıp kendime gülümsedim.",
    "Bulaşık makinesini boşaltmak dünyanın en sıkıcı işi olabilir.",
    "Çoraplarımın eşleri çamaşır makinesinde yine kaybolmuş.",
    "Bugün hiç tanımadığım birine yol tarif ettim.",
    "Cep telefonumun ekran koruyucusu sonunda çatladı.",
    "Yolda bulduğum bozuk parayı bir çocuğa verdim.",
    "Bugün gökyüzünde tek bir bulut bile yok.",
    "Radyoda çalan şarkıya bağıra bağıra eşlik ettim.",
    "Kahvemi yudumlarken bir yandan da haberleri okuyorum.",
    "Akşamları balkonda oturup serin rüzgarı hissetmek çok güzel.",
    "Yatağa girer girmez uyuyakalacak kadar yorgunum.",
    "Yarın sabah için kendime güzel bir kahvaltı hazırlayacağım.",
    "Bugün başardığım küçük şeyler için kendimle gurur duyuyorum.",
    "Her yeni gün, yeni bir şeyler öğrenmek için harika bir fırsat.",
    "Işıkları kapatıp uykuya dalmadan önce yarını düşündüm."
]


#preprocessing: metin verisini temizleme, küçük harfe çevirme, noktalama işaretlerini kaldırma gibi işlemler yapılır. Bu adım, modelin daha iyi öğrenmesini sağlar.
#kelimeleri sayilara cevirme (tokenization): Metin verisi, kelimeleri sayılara çevirerek modelin anlayabileceği bir formata dönüştürülür. Bu işlem için Tokenizer sınıfı kullanılabilir.

Tokenizer = Tokenizer()
Tokenizer.fit_on_texts(data)
total_words = len(Tokenizer.word_index) + 1
print("Toplam kelime sayısı:", total_words)


#n-gram dizileri olusturma(embedding): Metin verisi, n-gram dizilerine dönüştürülür. N-gram, metindeki kelimeleri n'li gruplara ayırarak oluşturulan dizilerdir.
#3-gram: kelimeleri 3'lü gruplara ayırarak diziler oluşturma Örneğin "ben yarin okula gidecegim" cümlesi için 3-gram dizileri: ["ben yarin okula", "yarin okula gidecegim"]
input_sequences = []
for line in data:
    token_list = Tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

print("Oluşturulan n-gram dizilerinin sayısı:", len(input_sequences))

#padding: Dizilerin uzunluklarını eşitlemek için padding işlemi yapılır. Bu, modelin girdi olarak alacağı dizilerin aynı uzunlukta olmasını sağlar.

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
print("Padding işleminden sonra dizilerin uzunluğu:", max_sequence_len)

#girdi ve hedef degiskenlerini ayiralım

X = input_sequences[:,:-1]#n-1 kelime girdi olarak kullanilir
y = input_sequences[:,-1]#n'inci kelime hedef degisken olarak kullanilir

#hedef degiskene one-hot encoding uygulayalim
y = tf.keras.utils.to_categorical(y, num_classes=total_words)
print(f"Hedef değişkenin one-hot encoding sonrası şekli: {y}")
"""
[123 456 789] -> [0, 0, 1, 0, 0, 1, 0, 0, 1] (örnek one-hot encoding)

"""

#LSTM training 
#lstm modelini olusturalim
model = Sequential()#modelimiz sequential yapida olacak, yani katmanlar birbirini takip edecek sekilde eklenecek
model.add(Embedding(total_words, 50, input_length=X.shape[1]))#embedding katmanı, kelimeleri düşük boyutlu vektörlere dönüştürür. total_words: kelime sayısı, 50: embedding boyutu, input_length: girdi dizisinin uzunluğu (n-1)
model.add(LSTM(100))#LSTM katmanı, sıralı verilerle çalışmak için kullanılır. 100: LSTM hücre sayısı
model.add(Dense(total_words, activation='softmax'))#Dense katmanı, tam bağlantılı bir katmandır ve çıkış boyutu total_words kadar olacaktır. activation='softmax' ifadesi, modelin her kelime için bir olasılık dağılımı üretmesini sağlar.


#modeli egitelim
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#modeli derlerken optimizer olarak 'adam' kullanılır, loss fonksiyonu olarak 'categorical_crossentropy' seçilir ve doğruluk metriği izlenir.
model.summary()#modelin özetini görüntüler

model.fit(X, y, epochs=100, verbose=1)
#model.fit() fonksiyonu, modeli eğitmek için kullanılır. X: girdi verisi, y: hedef değişken, epochs: eğitim döngüsü sayısı, verbose: eğitim sürecinin detaylarını gösterir.


#ornek cümleler uretelim
def generate_text(seed_text, next_words):#seed_text: başlangıç cümlesi, next_words: üretilecek kelime sayısı
    for _ in range(next_words):#next_words kadar kelime üretmek için döngü oluşturulur
        token_list = Tokenizer.texts_to_sequences([seed_text])[0]
        #seed_text'i token_list'e dönüştürür. Tokenizer.texts_to_sequences() fonksiyonu, verilen metni sayılara çevirir ve [0] ifadesi, tek bir cümle olduğu için ilk elemanı alır.
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        #token_list'i padding işlemiyle max_sequence_len-1 uzunluğuna getirir. max_sequence_len-1, modelin girdi olarak beklediği uzunluktur (n-1).
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]
        #model.predict() fonksiyonu, token_list'i girdi olarak alır ve
        predicted_word = Tokenizer.index_word[predicted_word_index]
        #predicted_word_index'e karşılık gelen kelimeyi Tokenizer.index_word sözlüğünden alır. Eğer index bulunamazsa boş string döner.
        seed_text += " " + predicted_word
        #seed_text'e tahmin edilen kelimeyi ekler.
    return seed_text

#ornek cümleler uretelim
print(generate_text("ben yarin", 5))#başlangıç cümlesi "ben yarin" ve 5 kelime üretilecek
print(generate_text("bugün hava", 5))#başlangıç cümlesi "bugün hava" ve 5 kelime üretilecek
print(generate_text("akşam yemeğinde",5))#başlangıç cümlesi "akşam yemeğinde" ve 5 kelime üretilecek

#ben yarin sonu için kendime gülümsedim çok
#bugün hava dünküne göre çok daha ılık
#akşam yemeğinde taze fasulye ve yanına pilav

        