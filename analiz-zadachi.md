# Анализ задачи

Как и в каждой главе, рассмотрим, что будет происходить в этом разделе. При передвижении робота по дому, он будет искать игрушки. Как только игрушка найдена, робот возьмет ее и отнесет к коробке с игрушками, а после уберет внутрь нее, отпустив игрушку. Затем робот снова двинется на поиск игрушек. На этом пути нужно будет избегать препятствий и опасностей, включая идущие вниз лестницы, которые могут повредить роботу.

Для первой части тестирования лестница будет прикрыта калиткой, а для второй части на лестнице будут находиться подушки. Роботу нет смысла биться о ступеньки, пока он находится в процессе обучения.

Начнем с предположения, что ничто в списке задач не требует от робота знания, где именно он находится. Правда ли это? Да, нам требуется лишь найти коробку с игрушками – вот, что важно. Можно ли найти коробку, не зная, где именно она находится? Ответ, конечно, заключается в том, что робот может просто искать коробку, пока не найдет ее. В предыдущей главе уже была разработана методика распознавания коробки с игрушками с помощью нейронной сети. 

Если бы робот выполнял куда более большую работу, например, чистил склад площадью в 1,000,000 квадратных футов, без карты было бы не обойтись. Но задача – очистить одну комнату размером 16х16. Время, потраченное на поиск коробки с игрушками, не так важно, учитывая, что мы не можем уехать далеко и нам в любом случае придется приехать к коробке. Так что поставим задачу обойтись в этом задании без создания карты. Итак, что еще потребуется?

* Перемещаться по комнате, избегая препятствия \(игрушки и мебель\) и опасности \(лестницы\)
* Найти игрушку
* Взять игрушку с помощью руки робота
* Отнести игрушку к коробке с игрушками
* Положить игрушку в коробку с игрушками
* Перемещаться дальше в поисках новой игрушки
* Если игрушек больше нет – остановиться

В других главах уже рассказывалось о том, как найти игрушку и взять ее в руки. В этой главе мы обсудим, как подъехать к игрушке, чтобы поднять ее. 

Я большой фанат фильма «Принцесса-невеста». Там есть бои на мечах, скалистые горы, борьба умов и гигантские грызуны. А так же есть урок планирования, который может нам помочь. Когда герои – великан Феззик, Иниго Монтойя и Уэстли – планируют штурмовать замок, чтобы спасти принцессу, первое, что спрашивает Уэстли: «Какие у нас пассивы? Какие у нас активы?» Итак:

* Наши пассивы: у нас есть маленький робот с очень ограниченными датчиками и вычислительной способностью. У нас есть комната, полная неубранных игрушек и несколько смертоносных лестниц, с которых робот может упасть.
* Наши активы: у нас есть робот с гусеницами, с помощью которых он передвигается, голос, одна камера и робо-рука. Робот имеет канал передачи данных через Wi-Fi к компьютеру. У нас есть эта книга и есть коробка для игрушек, которая имеет отличительный цвет. И есть много игрушек обычного размера.

Следующим шагом, независимо от того, разрабатываем ли мы роботов или захватываем замки, является мозговой штурм. Как же подойти к этой проблеме? Можно использовать SLAM и сделать карту, а затем найти робота на карте и использовать навигацию. Но чтобы в самом деле применить алгоритмы SLAM, потребуется больше сенсоров.

