# bmstu-ds-course
## Выпускная квалификационная работа по курсу "DataScience" образовательного центра МГТУ им. Н.Э. Баумана
## Слушатель: Тарасов Матвей Андреевич
### Тема
+ **Разработка тестового программного обеспечения, реализующего метод детектирования аварийной 
    ситуации по показаниям инерциальной навигационной системы в условиях отложенных данных с 
    использованием машинного обучения.**
### Задачи
+ **Проанализировать исходные данные**
+ **Извлечь признаки**
+ **Построить модели машинного обучения**
+ **Написать нейронную сеть**
+ **Выявить наиболее подходящую под решение задачи модель**
+ **Разработать приложение с графическим интерфейсом или интерфейсом командной строки,**
  **которое будет выдавать прогноз выбранной модели в зависимости от поступивших данных**
+ **Разместить результаты исcледования на GitHub**

> ### Описание
>
> В данной работе будет рассмотрен процесс разработки системы
детектирования аварийных ситуаций БПЛА мультироторного типа в условиях
отложенных данных с использованием машинного обучения на основе анализа
информации, получаемой от инерциальных датчиков. В частности, будет
проанализировано, как мониторинг вибрации может служить индикатором
возникновения потенциально опасных состояний. Вибрация является одним из
универсальных показателей, характеризующих работу механических систем, и
несет в себе информацию о состоянии узлов и агрегатов летательного аппарата.
Понимание ее режимов и закономерностей позволяет обнаруживать отклонения,
указывающие на возможные неисправности или непредвиденные внешние
воздействия.
>
>Ключевым моментом в исследовании будет являться реализация методов
обработки сигналов, получаемых от инерциальных датчиков, с целью создания
системы, способной реагировать на изменения в поведении БПЛА. Эта система
будет включать в себя алгоритмы детектирования аномалий, основанные на
машинном обучениии, которые позволят идентифицировать аварийные ситуации
до того, как они приведут к критическим последствиям. Таким образом, работа
не только внесет вклад в теорию и практику безопасности эксплуатации БПЛА,
но и откроет новые горизонты для их применения в различных областях, включая
экологический мониторинг, поисково-спасательные операции и др.
>
>Таким образом, разработка системы детектирования аварийных ситуаций
является важным шагом к повышению надежности и безопасности
мультироторных БПЛА. Успешная реализация данного проекта может привести
к значительным улучшениям как в области технологии управления БПЛА, так и
в области обеспечения безопасности полетов.

### Ход выполнения и результаты работы представлены в файлах:
+ [Notebook](https://github.com/amigoris/BMSTU_Diplom/Notebook_and_datasets/DroneFailureDetection.ipynb)
+ [Приложение](https://github.com/amigoris/BMSTU_Diplom/InterfaceFlask)
+ [Пояснительная записка](https://github.com/amigoris/BMSTU_Diplom/Documentation/Documentation.pdf)
+ [Презентация](https://github.com/amigoris/BMSTU_Diplom/Documentation/Presentation.pdf)
  
[Исходные данные](https://github.com/tiiuae/UAV-Realistic-Fault-Dataset/tree/main)