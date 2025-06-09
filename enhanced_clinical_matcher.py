"""
Enhanced Clinical Matcher for ICD-10 Code Ranking
This module provides improved TF-IDF matching specifically optimized for clinical text
and ICD-10 code matching in emergency department settings.
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class ClinicalTextPreprocessor:
    """
    Specialized preprocessor for clinical text to improve TF-IDF matching
    """
    
    def __init__(self):
        # Common clinical abbreviations that should be expanded
        self.abbreviations = {
            'cp': 'chest pain',
            'sob': 'shortness breath dyspnea',
            'doe': 'dyspnea exertion',
            'ekg': 'electrocardiogram',
            'ecg': 'electrocardiogram',
            'mi': 'myocardial infarction heart attack',
            'cad': 'coronary artery disease',
            'htn': 'hypertension',
            'dm': 'diabetes mellitus',
            'copd': 'chronic obstructive pulmonary disease',
            'chf': 'congestive heart failure',
            'cva': 'cerebrovascular accident stroke',
            'tia': 'transient ischemic attack',
            'gi': 'gastrointestinal',
            'uti': 'urinary tract infection',
            'uri': 'upper respiratory infection',
            'abd': 'abdominal abdomen',
            'hx': 'history',
            'sx': 'symptoms',
            'tx': 'treatment',
            'dx': 'diagnosis',
            'rx': 'prescription medication',
            'nsr': 'normal sinus rhythm',
            'afib': 'atrial fibrillation',
            'pvcs': 'premature ventricular contractions',
            'nstemi': 'non st elevation myocardial infarction',
            'stemi': 'st elevation myocardial infarction',
            'pe': 'pulmonary embolism',
            'dvt': 'deep vein thrombosis',
            'cxr': 'chest xray radiograph',
            'ct': 'computed tomography cat scan',
            'mri': 'magnetic resonance imaging',
            'ed': 'emergency department',
            'er': 'emergency room',
            'icu': 'intensive care unit',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'prn': 'as needed',
            'po': 'by mouth oral',
            'iv': 'intravenous',
            'im': 'intramuscular',
            'sq': 'subcutaneous',
            'wbc': 'white blood cell',
            'rbc': 'red blood cell',
            'hgb': 'hemoglobin',
            'plt': 'platelet',
            'bun': 'blood urea nitrogen',
            'cr': 'creatinine',
            'na': 'sodium',
            'k': 'potassium',
            'cl': 'chloride',
            'co2': 'carbon dioxide bicarbonate',
            'alt': 'alanine aminotransferase',
            'ast': 'aspartate aminotransferase',
            'alk': 'alkaline phosphatase'
        }
        
        # General medical term patterns - NO disease-specific expansions
        # This helps with term variations but doesn't hardcode specific conditions
        self.term_patterns = {
            # Anatomical variations
            r'\b(\w+)al\b': r'\1al \1',  # e.g., "pericardial" → "pericardial pericard"
            r'\b(\w+)ic\b': r'\1ic \1',  # e.g., "thoracic" → "thoracic thorac"
            r'\b(\w+)itis\b': r'\1itis \1 inflammation',  # e.g., "pericarditis" → "pericarditis pericard inflammation"
            r'\b(\w+)osis\b': r'\1osis \1',  # e.g., "fibrosis" → "fibrosis fibros"
            r'\b(\w+)emia\b': r'\1emia \1',  # e.g., "anemia" → "anemia anem"
        }
        
        # Clinical stopwords - words that appear everywhere and reduce specificity
        self.clinical_stopwords = {
            'patient', 'presents', 'reports', 'denies', 'states', 'history',
            'admission', 'discharge', 'hospital', 'clinic', 'emergency',
            'department', 'room', 'physician', 'nurse', 'staff', 'medical',
            'noted', 'shows', 'appears', 'found', 'seen', 'observed',
            'year', 'old', 'male', 'female', 'man', 'woman', 'today',
            'yesterday', 'days', 'weeks', 'months', 'time', 'times',
            'normal', 'stable', 'alert', 'oriented', 'cooperative',
            'review', 'systems', 'negative', 'positive', 'otherwise',
            'general', 'mild', 'moderate', 'severe', 'acute', 'chronic',
            'recent', 'onset', 'duration', 'associated', 'denies'
        }
    
    def preprocess_text(self, text):
        """
        Comprehensive preprocessing for clinical text
        """
        if pd.isna(text):
            return ''
        
        text = str(text).lower()
        
        # Step 1: Expand abbreviations
        for abbr, expansion in self.abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', expansion, text)
        
        # Step 2: Apply medical term patterns to extract root forms
        # This helps match "pericarditis" in note to "pericardial" in description
        for pattern, replacement in self.term_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # Step 3: Remove measurements and units but keep the context
        text = re.sub(r'\b\d+\s*(mg|ml|mcg|units?|%|mmhg|bpm|/min|cm|mm|degrees?|f|c)\b', ' ', text, flags=re.IGNORECASE)
        
        # Step 4: Keep medical terms with hyphens
        text = re.sub(r'([a-z])-([a-z])', r'\1\2', text)  # Remove hyphens except in medical terms
        
        # Step 5: Remove special characters but keep medical notation
        text = re.sub(r'[^a-zA-Z\s-]', ' ', text)
        
        # Step 6: Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_enhanced_vectorizer(self):
        """
        Create a TF-IDF vectorizer with clinical optimizations
        """
        # Combine standard English and clinical stopwords
        all_stopwords = list(ENGLISH_STOP_WORDS) + list(self.clinical_stopwords)
        
        return TfidfVectorizer(
            preprocessor=self.preprocess_text,
            tokenizer=None,  # Use default after preprocessing
            min_df=1,  # Don't exclude rare terms (important for specific conditions)
            max_df=0.90,  # Exclude terms in >90% of documents
            ngram_range=(1, 3),  # Include up to 3-grams for medical phrases
            stop_words=all_stopwords,
            sublinear_tf=True,  # Use log normalization
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            max_features=10000  # Limit vocabulary size
        )


class EnhancedClinicalMatcher:
    """
    Enhanced matching system with multiple strategies for ICD-10 code ranking
    """
    
    def __init__(self, codes_df):
        """
        Initialize the matcher with a codes dataframe
        
        Expected columns in codes_df:
        - ICD_10_CM_diagnosis_codes: The ICD-10 code
        - ED Short List Term: Primary description
        - description: Longer description
        - ED Short List Included: Additional included conditions
        """
        self.codes_df = codes_df.copy()
        self.preprocessor = ClinicalTextPreprocessor()
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Prepare the codes data
        self._prepare_codes_data()
    
    def _prepare_codes_data(self):
        """
        Prepare and enhance the codes data for better matching
        """
        print("Preparing codes data...")
        
        # Create combined text with field weighting
        self.codes_df['combined_text'] = (
            self.codes_df['ED Short List Term'].fillna('').astype(str) + ' ' +
            self.codes_df['ED Short List Term'].fillna('').astype(str) + ' ' +  # Double weight for term
            self.codes_df['description'].fillna('').astype(str) + ' ' +
            self.codes_df['ED Short List Included'].fillna('').astype(str)
        ).str.strip()
        
        # NO CODE-SPECIFIC ENHANCEMENTS - We want a general solution!
        
        # Build vectorizer and matrix
        print("Building TF-IDF matrix...")
        self.vectorizer = self.preprocessor.create_enhanced_vectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.codes_df['combined_text'])
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def rank_codes(self, clinical_note, method='hybrid', top_k=100, return_diagnostics=False):
        """
        Rank codes using specified method
        
        Args:
            clinical_note: The clinical note text
            method: 'tfidf', 'exact', or 'hybrid'
            top_k: Number of top codes to return
            return_diagnostics: Whether to return diagnostic information
        
        Returns:
            DataFrame with ranked codes
        """
        # Preprocess the clinical note
        processed_note = self.preprocessor.preprocess_text(clinical_note)
        
        if method == 'tfidf':
            scores = self._tfidf_ranking(processed_note)
        elif method == 'exact':
            scores = self._exact_match_ranking(clinical_note, processed_note)
        elif method == 'hybrid':
            tfidf_scores = self._tfidf_ranking(processed_note)
            exact_scores = self._exact_match_ranking(clinical_note, processed_note)
            # Combine scores with weights
            scores = 0.7 * tfidf_scores + 0.3 * exact_scores
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add scores to dataframe
        results_df = self.codes_df.copy()
        results_df['similarity_score'] = scores
        results_df = results_df.sort_values('similarity_score', ascending=False).head(top_k)
        
        if return_diagnostics:
            diagnostics = self._generate_diagnostics(clinical_note, processed_note, results_df)
            return results_df, diagnostics
        
        return results_df
    
    def _tfidf_ranking(self, processed_note):
        """Standard TF-IDF ranking"""
        query_vec = self.vectorizer.transform([processed_note])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return similarities
    
    def _exact_match_ranking(self, original_note, processed_note):
        """Boost codes with exact term matches - GENERAL approach"""
        scores = np.zeros(len(self.codes_df))
        
        # Extract significant medical terms from the note (3+ characters, not stopwords)
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        all_stopwords = set(ENGLISH_STOP_WORDS) | self.preprocessor.clinical_stopwords
        
        # Get words from original note
        note_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', original_note.lower()))
        note_words = note_words - all_stopwords
        
        # Also get medical terms that end in -itis, -osis, -emia, etc.
        medical_suffixes = ['itis', 'osis', 'emia', 'pathy', 'algia', 'ectomy', 'ostomy', 
                           'otomy', 'ology', 'plasty', 'scopy', 'graphy', 'gram']
        
        medical_terms = set()
        for word in note_words:
            for suffix in medical_suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    medical_terms.add(word)
                    # Also add the root (e.g., "pericarditis" → "pericard")
                    root = word[:-len(suffix)]
                    if len(root) > 2:
                        medical_terms.add(root)
        
        # Score based on matches
        for idx, row in self.codes_df.iterrows():
            code_text = (str(row['ED Short List Term']) + ' ' + 
                        str(row['description']) + ' ' + 
                        str(row['ED Short List Included'])).lower()
            
            # Count exact word matches
            code_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', code_text))
            
            # Regular word matches (weight = 1.0)
            common_words = note_words & code_words
            scores[idx] += len(common_words) * 1.0
            
            # Medical term matches (weight = 2.0) - these are more significant
            medical_matches = medical_terms & code_words
            scores[idx] += len(medical_matches) * 2.0
            
            # Bonus for matching terms in the primary field (ED Short List Term)
            if pd.notna(row['ED Short List Term']):
                term_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', row['ED Short List Term'].lower()))
                primary_matches = note_words & term_words
                scores[idx] += len(primary_matches) * 0.5  # Extra weight for primary field
            
            # Length-normalized matching: favor codes where a higher percentage of words match
            if len(code_words) > 0:
                match_ratio = len(common_words) / len(code_words)
                scores[idx] += match_ratio * 0.5
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def _generate_diagnostics(self, original_note, processed_note, results_df):
        """Generate diagnostic information for debugging"""
        diagnostics = {
            'original_note_length': len(original_note),
            'processed_note_length': len(processed_note),
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            'target_code_rank': None,
            'top_5_codes': [],
            'matching_terms': []
        }
        
        # Find pericarditis rank (as an example)
        pericard_mask = results_df['ICD_10_CM_diagnosis_codes'] == 'I30.9'
        if pericard_mask.any():
            rank = results_df.reset_index(drop=True).index[pericard_mask].tolist()
            diagnostics['target_code_rank'] = rank[0] + 1 if rank else None
        
        # Top 5 codes
        for idx, row in results_df.head(5).iterrows():
            diagnostics['top_5_codes'].append({
                'code': row['ICD_10_CM_diagnosis_codes'],
                'term': row['ED Short List Term'],
                'score': row['similarity_score']
            })
        
        # Analyze matching terms
        query_vec = self.vectorizer.transform([processed_note])
        feature_names = self.vectorizer.get_feature_names_out()
        query_features = query_vec.nonzero()[1]
        
        for idx in query_features[:10]:  # Top 10 terms
            diagnostics['matching_terms'].append({
                'term': feature_names[idx],
                'tfidf_score': query_vec[0, idx]
            })
        
        return diagnostics


# Utility functions for testing and integration
def test_enhanced_matcher(clinical_note, codes_df):
    """
    Test the enhanced matcher with a clinical note
    """
    print("=" * 80)
    print("TESTING ENHANCED CLINICAL MATCHER")
    print("=" * 80)
    
    # Initialize the matcher
    matcher = EnhancedClinicalMatcher(codes_df)
    
    # Test different methods
    methods = ['tfidf', 'exact', 'hybrid']
    
    for method in methods:
        print(f"\n{method.upper()} Method Results:")
        print("-" * 40)
        
        results, diagnostics = matcher.rank_codes(
            clinical_note, 
            method=method, 
            return_diagnostics=True
        )
        
        # Find pericarditis rank if it exists
        pericard_mask = results['ICD_10_CM_diagnosis_codes'].str.startswith('I30')
        if pericard_mask.any():
            pericard_rows = results[pericard_mask]
            for idx, row in pericard_rows.iterrows():
                rank = results.index.get_loc(idx) + 1
                print(f"  {row['ICD_10_CM_diagnosis_codes']} ({row['ED Short List Term']}) - Rank: #{rank}")
        
        print(f"\n  Top 5 codes:")
        for i, (idx, row) in enumerate(results.head(5).iterrows()):
            print(f"    {i+1}. {row['ICD_10_CM_diagnosis_codes']}: {row['ED Short List Term'][:50]} (score: {row['similarity_score']:.3f})")
        
        if method == 'hybrid':
            print(f"\n  Diagnostic Info:")
            print(f"    Vocabulary size: {diagnostics['vocabulary_size']}")
            print(f"    Matching terms: {[t['term'] for t in diagnostics['matching_terms'][:5]]}")
    
    return results


def compare_with_original(clinical_note, codes_df, original_ranking_function=None):
    """
    Compare enhanced matcher results with original TF-IDF ranking
    """
    print("\nCOMPARISON WITH ORIGINAL RANKING")
    print("=" * 80)
    
    # Get enhanced results
    matcher = EnhancedClinicalMatcher(codes_df)
    enhanced_results = matcher.rank_codes(clinical_note, method='hybrid')
    
    # If original ranking function provided, compare
    if original_ranking_function:
        original_results = original_ranking_function(clinical_note, codes_df)
        
        # Find specific codes in both rankings
        test_codes = ['I30.9', 'R07.9', 'I21.9', 'J44.0']
        
        print("\nCode Ranking Comparison:")
        print(f"{'Code':<10} {'Description':<40} {'Original':<10} {'Enhanced':<10}")
        print("-" * 80)
        
        for code in test_codes:
            orig_mask = original_results['ICD_10_CM_diagnosis_codes'] == code
            enh_mask = enhanced_results['ICD_10_CM_diagnosis_codes'] == code
            
            if orig_mask.any() and enh_mask.any():
                orig_rank = original_results.index[orig_mask].tolist()[0] + 1
                enh_rank = enhanced_results.index[enh_mask].tolist()[0] + 1
                desc = enhanced_results[enh_mask]['ED Short List Term'].iloc[0][:40]
                
                print(f"{code:<10} {desc:<40} #{orig_rank:<9} #{enh_rank:<9}")
    
    return enhanced_results


# Example usage for Streamlit integration
def integrate_with_streamlit(clinical_note, codes_df):
    """
    Example of how to integrate with existing Streamlit app
    """
    # Initialize matcher (could cache this)
    matcher = EnhancedClinicalMatcher(codes_df)
    
    # Get rankings
    results = matcher.rank_codes(clinical_note, method='hybrid', top_k=100)
    
    # Return in same format as original function
    return results[['ICD_10_CM_diagnosis_codes', 'ED Short List Term', 'similarity_score']]


if __name__ == "__main__":
    # Example test
    print("Enhanced Clinical Matcher Module Loaded Successfully!")
    print("\nTo use in your code:")
    print("1. Import: from enhanced_clinical_matcher import EnhancedClinicalMatcher")
    print("2. Initialize: matcher = EnhancedClinicalMatcher(your_codes_df)")
    print("3. Rank: results = matcher.rank_codes(clinical_note, method='hybrid')")