classpath = .:../..:../lib/commons-math3-3.2.jar
scalac += -cp $(classpath) -d ../..

.PHONY: clean prob

clean:
	-rm *.class

prob: ArrayDist.scala              	\
			ContinuousDistribution.scala  \
			DirichletHashMapDist.scala   	\
			DiscreteDistribution.scala   	\
			Distribution.scala           	\
			HashMapDist.scala            	\
			MultivarNormalDist.scala
	$(scalac)                       \
		Distribution.scala            \
		ContinuousDistribution.scala  \
		DiscreteDistribution.scala    \
		ArrayDist.scala               \
		HashMapDist.scala             \
		DirichletHashMapDist.scala    \
		MultivarNormalDist.scala