package org.opencog.vqa.relex;

import relex.feature.FeatureNode;
import relex.feature.RelationCallback;

class YesNoPredadjToSchemeQueryConverter implements ToQueryConverter {

    @Override
    public boolean isApplicable(RelexFormula formula) {
        return formula.getFullFormula().equals(this.getFullFormula());
    }
    
    @Override
    public String getFullFormula() {
    	return "_predadj(A, B)";
    }
    
    @Override
    public String getQuestionType() {
    	return "yes/no";
    }

    private String getAndLink() {
        return "  (AndLink\n" +
        "    (InheritanceLink (VariableNode \"$X\") (ConceptNode \"BoundingBox\"))\n" +
        "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"%1$s\")) )\n" +
        "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"%2$s\")) )\n" +
        "  )\n";
    }

    @Override
    public String getSchemeQuery(RelexFormula relexFormula) {
        RelexVisitor visitor = new RelexVisitor();
        relexFormula.getRelexSentence().foreach(visitor);
        // $X - is a bounding box
        // visitor.object - object which is supposed to be in some state
        // visitor.state - the question is whether object is in this state
        return String.format("(BindLink\n" +
                "  (TypedVariableLink (VariableNode \"$X\") (TypeNode \"ConceptNode\"))\n" +
                this.getAndLink() +
                this.getAndLink() +
                ")\n", visitor.object, visitor.state);
    }

    @Override
    public String getSchemeQueryURE(RelexFormula relexFormula) {
        RelexVisitor visitor = new RelexVisitor();
        relexFormula.getRelexSentence().foreach(visitor);
        // $X - is a bounding box
        // visitor.object - object which is supposed to be in some state
        // visitor.state - the question is whether object is in this state
        return String.format("(conj-bc (AndLink\n" +
                "    (InheritanceLink (VariableNode \"$X\") (ConceptNode \"BoundingBox\"))\n" +
                "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"%1$s\")) )\n" +
                "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"%2$s\")) )\n" +
                "  )\n )", visitor.object, visitor.state);
    }

    @Override
    public String getSchemeQueryPM(RelexFormula relexFormula) {
		return "(cog-execute! " + this.getSchemeQuery(relexFormula) + ")";
    }

    private static class RelexVisitor implements RelationCallback {
        
        String object;
        String state;
        
        @Override
        public Boolean UnaryRelationCB(FeatureNode node, String attrName) {
            return Boolean.FALSE;
        }

        @Override
        public Boolean BinaryRelationCB(String relation, FeatureNode srcNode, FeatureNode tgtNode) {
            if (relation.equals("_predadj")) {
                object = RelexUtils.getFeatureNodeName(srcNode);
                state = RelexUtils.getFeatureNodeName(tgtNode);
            }
            return Boolean.FALSE;
        }

        @Override
        public Boolean BinaryHeadCB(FeatureNode from) {
            return Boolean.FALSE;
        }
    }
    
}
