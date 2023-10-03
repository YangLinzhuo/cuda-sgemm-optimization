for ((i=256; i <= 6400; i+=256))
do
	./sgemm 0 $i
done

