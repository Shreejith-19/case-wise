export const JURISDICTIONS = [
  'All',
  'Central',
  'Andaman and Nicobar Islands',
  'Andhra Pradesh',
  'Arunachal Pradesh',
  'Assam',
  'Bihar',
  'Chandigarh',
  'Chhattisgarh',
  'Dadra and Nagar Haveli and Daman and Diu',
  'Delhi',
  'Goa',
  'Gujarat',
  'Haryana',
  'Himachal Pradesh',
  'Jammu and Kashmir',
  'Jharkhand',
  'Karnataka',
  'Kerala',
  'Ladakh',
  'Madhya Pradesh',
  'Maharashtra',
  'Manipur',
  'Meghalaya',
  'Mizoram',
  'Nagaland',
  'Odisha',
  'Puducherry',
  'Punjab',
  'Rajasthan',
  'Sikkim',
  'Tamil Nadu',
  'Telangana',
  'Tripura',
  'Uttar Pradesh',
  'Uttarakhand',
  'West Bengal',
];

export const DOC_TYPES = ['All', 'Central Act', 'State Act', 'Rule', 'Regulation'];

export const ABSTAIN_MESSAGES = {
  missing_provision: {
    title: 'Section or Rule Not Found',
    detail: 'The Act was located, but the specific section or rule referenced does not appear in the statute.',
  },
  unresolved_law: {
    title: 'Statute Not Found',
    detail:
      'This law could not be found in Indian Central or State statutes, rules, or regulations. Case law and foreign legislation are not included.',
  },
  low_rerank_score: {
    title: 'No Relevant Provision Found',
    detail: 'No statutory provision matched your query with sufficient confidence to provide a reliable answer.',
  },
  llm: {
    title: 'Insufficient Statutory Basis',
    detail: 'The retrieved statutory text does not directly answer the question asked.',
  },
};
